# -*- coding: utf-8 -*-
"""Sinh nb_qwen_seqbias_posthoc.ipynb — KHONG FINE-TUNE LAI. Nap adapter LoRA DA
CO SAN (tai len tu may local ban da tai ve) roi dieu chinh SEQUENCE BIAS luc sinh
van ban de nang recall lop thieu so, dua tren tap VAL de do gia tri bias, roi danh
gia tren TOAN BO test set.

GIOI HAN QUAN TRONG (da kiem chung bang tokenizer that): SequenceBiasLogitsProcessor
chi cong bias vao TOKEN CUOI khi cac token truoc DA KHOP -> chi hieu qua that su cho
DAP AN NGAN (1-2 token). Vi vay CHI ap dung cho Q9/Q10 (yes/no), KHONG ap dung cho
Q2/Q8 (cum tu dai 5-7 token, bias se gan nhu vo tac dung o dau cau)."""
import os
from _mknb import make_notebook
HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPT = r'''# ============================================================
# Qwen2.5-VL-3B + LoRA (KHONG train lai) — SEQUENCE BIAS luc sinh cho Q9/Q10
# CHAY: python qwen_seqbias_eval.py   (single-GPU, chi suy luan, khong can torchrun)
# ============================================================
import os, re, glob, sys, time, json
import numpy as np, torch, pandas as pd
from PIL import Image
from collections import Counter
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from peft import PeftModel
import warnings; warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ── Duong dan du lieu (tu dong do) ────────────────────────────
csv_train = glob.glob("/kaggle/input/**/train.csv", recursive=True)
imgs_any = (glob.glob("/kaggle/input/**/*.jpg", recursive=True)+
            glob.glob("/kaggle/input/**/*.png", recursive=True)+
            glob.glob("/kaggle/input/**/*.jpeg", recursive=True))
QA_DIR = os.path.dirname(csv_train[0])
IMG_DIR = Counter(os.path.dirname(p) for p in imgs_any).most_common(1)[0][0]
imgmap = {os.path.basename(p): p for p in glob.glob(os.path.join(IMG_DIR, "*"))}

# ── Tim adapter LoRA da upload (bat ky ten dataset nao chua adapter_config.json) ──
adapter_hits = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
assert adapter_hits, ("KHONG tim thay adapter_config.json trong /kaggle/input — "
    "ban can upload thu muc qwen_lora (da tai ve tu lan fine-tune truoc) thanh 1 "
    "Kaggle Dataset moi va Add vao notebook nay truoc khi chay.")
ADAPTER_DIR = os.path.dirname(adapter_hits[0])
print("Da tim thay adapter tai:", ADAPTER_DIR)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 64*28*28; MAX_PIXELS = 256*28*28
OUT_CSV_BASE  = "/kaggle/working/qwen_seqbias_base_raw.csv"
OUT_CSV_BIAS  = "/kaggle/working/qwen_seqbias_tuned_raw.csv"
METRIC_CSV    = "/kaggle/working/qwen_seqbias_metrics.csv"

SYSTEM_PROMPT = ("Bạn là trợ lý phân tích ảnh vi phạm sử dụng vỉa hè. "
    "Trả lời NGẮN GỌN, CHỈ dùng đúng các từ có trong lựa chọn của câu hỏi. "
    "Nếu có nhiều đáp án đúng, nối bằng ' và '. Không giải thích thêm.")

QID_ORDER=["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10"]
WHAT_QIDS={"Q2","Q5","Q8"}; DERIVED_BIN={"Q4"}
def normalize(s): s=str(s).strip().lower(); s=re.sub(r"\s*/\s*","/",s); return re.sub(r"\s+"," ",s)
def atomic_set(s): return set(t.strip() for t in normalize(s).split(" và ") if t.strip())
def to_binary(a): return "không" if normalize(a) in {"không","không có"} else "có"
def answer_atoms(qid,ans): return {to_binary(ans)} if qid in DERIVED_BIN else atomic_set(ans)
def build_vocab(df):
    vocab=set(); cand={q:set() for q in QID_ORDER}
    for _,r in df.iterrows():
        q=str(r["qid"]); ats=answer_atoms(q,r["answer"]); vocab|=ats; cand[q]|=ats
    vocab=sorted(vocab); idx={a:i for i,a in enumerate(vocab)}
    return vocab, idx, {q:sorted(idx[a] for a in cand[q]) for q in QID_ORDER if cand[q]}

train_df=pd.read_csv(os.path.join(QA_DIR,"train.csv")); train_df["qid"]=train_df["qid"].astype(str)
val_df=pd.read_csv(os.path.join(QA_DIR,"val.csv")); val_df["qid"]=val_df["qid"].astype(str)
test_df=pd.read_csv(os.path.join(QA_DIR,"test.csv")); test_df["qid"]=test_df["qid"].astype(str)
vocab, idx, cand = build_vocab(pd.concat([train_df,val_df,test_df],ignore_index=True))
q_text = {q: pd.concat([train_df,val_df,test_df])[lambda d:d["qid"]==q]["question"].mode().iloc[0] for q in QID_ORDER}

def parse_single_atom(raw_text,qid):
    t=normalize(raw_text); cds=sorted((vocab[i] for i in cand[qid]),key=len,reverse=True)
    found=[]
    for c in cds:
        if c in t and not any(c in f for f in found): found.append(c)
    if not found: return cds[-1]
    neg=[f for f in found if f.startswith("không")]
    return neg[0] if neg else found[0]
def parse_multi_atoms(raw_text,qid):
    t=normalize(raw_text); cds=sorted((vocab[i] for i in cand[qid]),key=len,reverse=True)
    found=[]
    for c in cds:
        if c in t and not any(c in f for f in found): found.append(c)
    return found if found else [cds[-1]]

def build_messages(image, question):
    return [{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":question}]}]

# ── Nap base model 4-bit + gan adapter (KHONG train, chi eval) ──
print("Nap base model + adapter...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
processor.tokenizer.padding_side = "left"
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=bnb, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()
for _k in ("temperature","top_p","top_k"):
    if hasattr(model.generation_config, _k): setattr(model.generation_config, _k, None)
model.generation_config.do_sample = False
print("Da nap xong (khong fine-tune lai).")

@torch.no_grad()
def generate_answer(image_path, question, sequence_bias=None):
    msgs = build_messages(image_path, question)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs,_ = process_vision_info(msgs)
    enc = processor(text=[text], images=imgs, return_tensors="pt").to(model.device)
    gen_kwargs = dict(max_new_tokens=24, do_sample=False)
    if sequence_bias: gen_kwargs["sequence_bias"] = sequence_bias
    out = model.generate(**enc, **gen_kwargs)
    gen = out[:, enc["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

# ── PHUONG PHAP: sequence-bias CHI cho Q9/Q10 (dap an ngan 1-2 token) ──
# Cac cum tu dai (Q2/Q5/Q8) KHONG dung duoc: bias chi cong vao token CUOI khi cac
# token truoc da khop -> voi cum 5-7 token gan nhu vo tac dung luc bat dau sinh.
BIAS_QIDS = ["Q9", "Q10"]
MINORITY = {"Q9": "có", "Q10": "không"}

def token_ids(word):
    return tuple(processor.tokenizer(word, add_special_tokens=False)["input_ids"])

def eval_subset(df, q, sequence_bias=None, tag=""):
    sub = df[df["qid"]==q]
    rows=[]
    from tqdm.auto import tqdm
    for _, r in tqdm(sub.iterrows(), total=len(sub), desc=f"{tag} {q}", ncols=90):
        p = imgmap.get(r["filename"])
        if p is None: continue
        raw = generate_answer(p, q_text[q], sequence_bias=sequence_bias)
        pred = parse_single_atom(raw, q)
        rows.append({"qid":q,"filename":r["filename"],"raw_answer":raw,"pred_atoms":pred,"gt_answer":r["answer"]})
    return pd.DataFrame(rows)

from sklearn.metrics import recall_score, f1_score, accuracy_score
def minority_recall(df, q, minority_label):
    gt=[to_binary(a) for a in df["gt_answer"]]; pr=list(df["pred_atoms"])
    yb=[1 if g==minority_label else 0 for g in gt]; pb=[1 if p==minority_label else 0 for p in pr]
    return recall_score(yb,pb,zero_division=0), accuracy_score(gt,pr)

def score(df):
    wf=[]; sf=[]; se=[]
    for q in QID_ORDER:
        sub=df[df["qid"]==q]
        if len(sub)==0: continue
        cds=[vocab[i] for i in cand[q]]
        gts=[answer_atoms(q,a) for a in sub["gt_answer"]]
        if q in WHAT_QIDS:
            prs=[set(str(a).split(" và ")) if pd.notna(a) else set() for a in sub["pred_atoms"]]
            Yt=np.array([[1 if c in g else 0 for c in cds] for g in gts])
            Yp=np.array([[1 if c in p else 0 for c in cds] for p in prs])
            wf.append(f1_score(Yt,Yp,average="macro",zero_division=0))
        else:
            gt=[list(g)[0] if g else cds[-1] for g in gts]; pr=list(sub["pred_atoms"])
            sf.append(f1_score(gt,pr,average="macro",labels=cds,zero_division=0)); se.append(accuracy_score(gt,pr))
    return (float(np.mean(wf)) if wf else 0, float(np.mean(sf)) if sf else 0, float(np.mean(se)) if se else 0)

# ── Buoc 1: do bias tot nhat tren VAL (grid nho) ──
print("\\n=== Do bias toi uu tren tap VAL (Q9, Q10) ===")
BIAS_GRID = [0.0, 1.0, 2.0, 3.0, 4.0]
best_bias = {}
for q in BIAS_QIDS:
    minority = MINORITY[q]
    ids = token_ids(minority)
    best_score, best_b = -1, 0.0
    for b in BIAS_GRID:
        sb = {ids: b} if b > 0 else None
        df_val = eval_subset(val_df, q, sequence_bias=sb, tag=f"val b={b}")
        rec, acc = minority_recall(df_val, q, minority)
        # diem ket hop: uu tien recall lop thieu so nhung khong danh doi acc qua nhieu
        combo = rec - max(0, (0.90-acc))  # phat neu acc tut duoi 90%
        print(f"  {q} bias={b:.1f}: recall({minority!r})={rec:.3f} EM={acc:.3f} combo={combo:.3f}")
        if combo > best_score: best_score, best_b = combo, b
    best_bias[q] = best_b
    print(f"  -> Chon bias={best_b} cho {q}")
json.dump(best_bias, open("/kaggle/working/qwen_seqbias_best.json","w"))

# ── Buoc 2: danh gia TOAN BO test set — KHONG bias (base) vs CO bias (tuned) ──
print("\\n=== Danh gia TOAN BO test set ===")
rows_base=[]; rows_tuned=[]
for q in QID_ORDER:
    sub = test_df[test_df["qid"]==q]
    sb = None
    if q in BIAS_QIDS and best_bias.get(q,0)>0:
        sb = {token_ids(MINORITY[q]): best_bias[q]}
    for _, r in sub.iterrows():
        p = imgmap.get(r["filename"])
        if p is None: continue
        raw_b = generate_answer(p, q_text[q], sequence_bias=None)
        pred_b = parse_multi_atoms(raw_b,q) if q in WHAT_QIDS else parse_single_atom(raw_b,q)
        pred_b = " và ".join(pred_b) if q in WHAT_QIDS else pred_b
        rows_base.append({"qid":q,"filename":r["filename"],"raw_answer":raw_b,"pred_atoms":pred_b,"gt_answer":r["answer"]})
        if q in BIAS_QIDS:
            raw_t = generate_answer(p, q_text[q], sequence_bias=sb)
            pred_t = parse_single_atom(raw_t,q)
            rows_tuned.append({"qid":q,"filename":r["filename"],"raw_answer":raw_t,"pred_atoms":pred_t,"gt_answer":r["answer"]})
        else:
            rows_tuned.append(rows_base[-1])  # cau khac Q9/Q10: giu nguyen (khong bias)

df_base = pd.DataFrame(rows_base); df_tuned = pd.DataFrame(rows_tuned)
df_base.to_csv(OUT_CSV_BASE, index=False); df_tuned.to_csv(OUT_CSV_BIAS, index=False)

wF0,sF0,sEM0 = score(df_base)
wF1,sF1,sEM1 = score(df_tuned)
metrics = pd.DataFrame([
    {"mô hình":"Qwen adapter (khong bias, tai lap)","what_F1macro":round(wF0,4),"single_F1macro":round(sF0,4),"single_EM":round(sEM0,4)},
    {"mô hình":"Qwen adapter + sequence-bias (Q9,Q10)","what_F1macro":round(wF1,4),"single_F1macro":round(sF1,4),"single_EM":round(sEM1,4)},
])
metrics.to_csv(METRIC_CSV, index=False)
print(metrics.to_string(index=False))

print("\\nRecall lop thieu so — khong bias vs co bias:")
for q in BIAS_QIDS:
    m = MINORITY[q]
    r0,_ = minority_recall(df_base[df_base["qid"]==q], q, m)
    r1,_ = minority_recall(df_tuned[df_tuned["qid"]==q], q, m)
    print(f"  {q} lop {m!r}: {r0:.3f} -> {r1:.3f}  (bias da chon={best_bias.get(q,0)})")
print(f"\\nDa luu: {OUT_CSV_BASE}, {OUT_CSV_BIAS}, {METRIC_CSV}")
'''

cells = [
("markdown", """# Sequence-Bias hậu kiểm cho Qwen2.5-VL-3B (KHÔNG fine-tune lại)

**Hạn chế nhắm tới**: Q9 "có" (hư hỏng bề mặt, recall 30%), Q10 "không" (lối đi bị
chặn, recall 53%) — model bỏ sót lớp thiểu số ngay cả sau fine-tune.

**Phương pháp: Sequence-bias lúc sinh** (`generate(..., sequence_bias=...)`) —
cộng thêm điểm ưu tiên cho token của đáp án thiểu số, **không cần huấn luyện lại**,
chỉ chỉnh lúc suy luận trên **adapter đã có sẵn**. Giá trị bias được **dò trên tập
val** (không đoán mò).

**Lưu ý**: bước đánh giá "Adapter (không bias, tái lập)" ở mục 3 chính là **fine-tune
baseline chạy trên TOÀN BỘ test set** (không phải mẫu 150/câu như lần đo gốc) — dùng
số này làm mốc full-test đáng tin cậy hơn để so với các phương pháp cải thiện khác.

> ⚠️ **Giới hạn kỹ thuật đã kiểm chứng**: `SequenceBiasLogitsProcessor` chỉ cộng
> bias vào **token cuối cùng khi các token trước đã khớp đúng chuỗi**. Với đáp án
> ngắn 1-2 token (Q9 "có"=1 token, Q10 "không"=2 token) → hiệu quả thật. Với cụm từ
> dài 5-7 token (Q2/Q5/Q8) → bias gần như vô tác dụng vì phải "đang gõ đúng cả cụm"
> mới kích hoạt. Vì vậy **notebook này CHỈ áp dụng cho Q9, Q10**.

## ⚠️ Cần làm trước khi chạy: upload adapter đã tải về

Notebook này **không train**, chỉ nạp adapter LoRA đã fine-tune từ trước. Bạn cần:
1. Vào Kaggle → **Datasets → New Dataset** → upload thư mục `qwen_lora` (đã tải về
   máy từ lần fine-tune baseline) → Create.
2. Trong notebook này, **Add Input** → chọn dataset adapter vừa tạo (thêm vào CÙNG
   notebook với dataset ảnh+CSV gốc).
3. Notebook tự dò `adapter_config.json` trong `/kaggle/input/**` — không cần sửa path.

> Chỉ cần **GPU T4 ×1** (không cần 2 — đây là suy luận, không train, không cần DDP).
> Nhanh hơn nhiều các notebook fine-tune khác vì không có bước huấn luyện."""),

("code", """# ===== 0. Cai thu vien =====
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","-U",
    "transformers>=4.49.0","accelerate>=0.34.0","qwen-vl-utils",
    "bitsandbytes>=0.46.1","peft","tqdm"], check=False)
print("Da cai xong thu vien.")"""),

("code", "%%writefile qwen_seqbias_eval.py\n" + SCRIPT),

("markdown", """## 2. Chạy (single-GPU, không cần torchrun vì chỉ suy luận)"""),

("code", """import subprocess as _sp, os as _os, time as _t2, torch
try:
    q=_sp.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],capture_output=True,text=True)
    mypid=_os.getpid(); stray=sorted({int(x) for x in q.stdout.split() if x.strip().isdigit()}-{mypid})
    for pid in stray:
        try: _os.kill(pid,9); print("Da kill GPU pid mo coi:",pid)
        except Exception: pass
    if stray: _t2.sleep(4)
except Exception as e:
    print("(bo qua don:",e,")")
_env={**_os.environ,"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}
print("Chay qwen_seqbias_eval.py ...", flush=True)
ret=_sp.run(["python","qwen_seqbias_eval.py"], env=_env)
print("Exit code:", ret.returncode)"""),

("markdown", """## 3. Xem kết quả

Dòng "Adapter (không bias, tái lập)" = **fine-tune baseline trên TOÀN BỘ test set**
(3891 mẫu) — mốc so sánh full-test đáng tin cậy hơn mẫu 150/câu ban đầu."""),

("code", """import pandas as pd
print(pd.read_csv("/kaggle/working/qwen_seqbias_metrics.csv").to_string(index=False))"""),

("markdown", """## 4. Đóng gói kết quả để tải về"""),

("code", """import os, glob, zipfile
from IPython.display import FileLink
OUT="/kaggle/working/qwen_seqbias_ket_qua.zip"
if os.path.exists(OUT): os.remove(OUT)
with zipfile.ZipFile(OUT,"w",zipfile.ZIP_DEFLATED) as zf:
    for f in glob.glob("/kaggle/working/qwen_seqbias_*.csv") + glob.glob("/kaggle/working/qwen_seqbias_*.json"):
        zf.write(f, os.path.basename(f)); print("+",os.path.basename(f))
    zf.write("qwen_seqbias_eval.py","qwen_seqbias_eval.py")
print(f"\\nDa tao {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
FileLink("qwen_seqbias_ket_qua.zip")"""),
]
make_notebook(cells, os.path.join(HERE, "nb_qwen_seqbias_posthoc.ipynb"))

# -*- coding: utf-8 -*-
"""Sinh nb_qwen_zeroshot_fulltest.ipynb — ZERO-SHOT Qwen2.5-VL-3B-Instruct tren
TOAN BO test set (3891 mau, khong sample 150/cau nhu ban cu). Doc lap, KHONG fine-
tune, 1 GPU T4, co checkpoint moi 200 anh de an toan neu bi ngat giua chung."""
import os
from _mknb import make_notebook
HERE = os.path.dirname(os.path.abspath(__file__))

cells = [
("markdown", """# Qwen2.5-VL-3B-Instruct — Zero-shot trên TOÀN BỘ test set (3.891 mẫu)

Bản trước chỉ lấy mẫu tối đa 150 ảnh/câu (~1.416 dòng) để so sánh nhanh. Notebook
này chạy **toàn bộ 3.891 mẫu test**, cùng chuẩn chấm điểm per-question atomic, để
có số liệu zero-shot đáng tin cậy đối chiếu với bản fine-tune (cũng đo trên toàn
bộ test set).

Có checkpoint mỗi 200 ảnh — nếu phiên bị ngắt, chạy lại sẽ tự tiếp tục.

> ⚠️ Bật **Internet** + **GPU T4** (chỉ cần 1 GPU, không train)."""),

("code", """# ===== 0. Cai thu vien =====
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","-U",
    "transformers>=4.49.0","accelerate>=0.34.0","qwen-vl-utils","tqdm"], check=False)
print("Da cai xong thu vien.")"""),

("code", """# ===== 1. Cau hinh & do duong dan Kaggle =====
import os, re, glob, time
import numpy as np, pandas as pd
import torch
from collections import Counter
SEED=42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, "| GPU:", torch.cuda.get_device_name(0) if DEVICE=="cuda" else "-")

csv_train = glob.glob("/kaggle/input/**/train.csv", recursive=True)
imgs_any  = (glob.glob("/kaggle/input/**/*.jpg", recursive=True) +
             glob.glob("/kaggle/input/**/*.png", recursive=True) +
             glob.glob("/kaggle/input/**/*.jpeg", recursive=True))
assert csv_train, "Khong tim thay train.csv trong /kaggle/input"
assert imgs_any,  "Khong tim thay anh trong /kaggle/input"
QA_DIR  = os.path.dirname(csv_train[0])
IMG_DIR = Counter(os.path.dirname(p) for p in imgs_any).most_common(1)[0][0]
imgmap  = {os.path.basename(p): p for p in glob.glob(os.path.join(IMG_DIR, "*"))}
print("QA_DIR :", QA_DIR); print("IMG_DIR:", IMG_DIR)"""),

("code", """# ===== 2. Nap annotation + xay GLOBAL ATOM VOCAB (dung chuan toan du an) =====
train = pd.read_csv(os.path.join(QA_DIR,"train.csv")); train["qid"]=train["qid"].astype(str)
val   = pd.read_csv(os.path.join(QA_DIR,"val.csv"));   val["qid"]=val["qid"].astype(str)
test  = pd.read_csv(os.path.join(QA_DIR,"test.csv"));  test["qid"]=test["qid"].astype(str)

QID_ORDER=["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10"]
WHAT_QIDS={"Q2","Q5","Q8"}; DERIVED_BIN={"Q4"}
def normalize(s): s=str(s).strip().lower(); s=re.sub(r"\\s*/\\s*","/",s); return re.sub(r"\\s+"," ",s)
def atomic_set(s): return set(t.strip() for t in normalize(s).split(" và ") if t.strip())
def to_binary(a): return "không" if normalize(a) in {"không","không có"} else "có"
def answer_atoms(qid,ans): return {to_binary(ans)} if qid in DERIVED_BIN else atomic_set(ans)

def build_vocab(df):
    vocab=set(); cand={q:set() for q in QID_ORDER}
    for _,r in df.iterrows():
        q=str(r["qid"]); ats=answer_atoms(q,r["answer"]); vocab|=ats; cand[q]|=ats
    vocab=sorted(vocab); idx={a:i for i,a in enumerate(vocab)}
    cand_idx={q:sorted(idx[a] for a in cand[q]) for q in QID_ORDER if cand[q]}
    return vocab, idx, cand_idx
vocab, idx, cand = build_vocab(pd.concat([train,val,test],ignore_index=True))
q_text = {q: pd.concat([train,val,test])[lambda d:d["qid"]==q]["question"].mode().iloc[0] for q in QID_ORDER}
print("GLOBAL atom vocab:", vocab)"""),

("markdown", """## 3. Nạp Qwen2.5-VL-3B-Instruct (bf16, 1 GPU — không train nên không cần lượng tử hóa)"""),

("code", """from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 64*28*28
MAX_PIXELS = 256*28*28   # giong cau hinh fine-tune -> so sanh cong bang ve do phan giai anh

processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
_dmap = {"": "cuda:0"} if DEVICE == "cuda" else None
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16 if DEVICE=="cuda" else torch.float32,
    device_map=_dmap)
model.eval()
processor.tokenizer.padding_side = "left"
for _k in ("temperature","top_p","top_k"):
    if hasattr(model.generation_config, _k): setattr(model.generation_config, _k, None)
model.generation_config.do_sample = False
print("Da nap", MODEL_ID)"""),

("markdown", """## 4. Suy luận zero-shot trên TOÀN BỘ test set (3.891 mẫu, có checkpoint)"""),

("code", """from tqdm.auto import tqdm
MAX_NEW_TOKENS = 24
CKPT = "/kaggle/working/qwen_zeroshot_fulltest_raw.csv"

SYSTEM_PROMPT = ("Bạn là trợ lý phân tích ảnh vi phạm sử dụng vỉa hè. "
    "Trả lời NGẮN GỌN, CHỈ dùng đúng các từ có trong lựa chọn của câu hỏi. "
    "Nếu có nhiều đáp án đúng, nối bằng ' và '. Không giải thích thêm.")

def build_messages(image_path, question):
    return [{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":[{"type":"image","image":image_path},{"type":"text","text":question}]}]

def parse_multi_atoms(raw_text, qid):
    t = normalize(raw_text)
    cds = sorted((vocab[i] for i in cand[qid]), key=len, reverse=True)
    found=[]
    for c in cds:
        if c in t and not any(c in f for f in found): found.append(c)
    return found if found else [cds[-1]]

def parse_single_atom(raw_text, qid):
    t = normalize(raw_text)
    cds = sorted((vocab[i] for i in cand[qid]), key=len, reverse=True)
    found=[]
    for c in cds:
        if c in t and not any(c in f for f in found): found.append(c)
    if not found: return cds[-1]
    neg = [f for f in found if f.startswith("không")]
    return neg[0] if neg else found[0]

@torch.no_grad()
def generate_answer(image_path, question):
    msgs = build_messages(image_path, question)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs,_ = process_vision_info(msgs)
    enc = processor(text=[text], images=imgs, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    gen = out[:, enc["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

worklist=[]
for q in QID_ORDER:
    sub = test[test["qid"]==q]   # TOAN BO, khong sample
    for _, r in sub.iterrows():
        if imgmap.get(r["filename"]) is not None:
            worklist.append((q, r["filename"], r["answer"]))
print(f"Tong so mau se chay (toan bo test): {len(worklist)}")

rows=[]; done=set()
if os.path.exists(CKPT):
    prev = pd.read_csv(CKPT); rows = prev.to_dict("records")
    done = {(str(x["qid"]), x["filename"]) for x in rows}
    print(f"Resume: da co {len(done)} ket qua tu checkpoint truoc.", flush=True)

import datetime as _dt

# QUAN TRONG cho "Save & Run All": tqdm dung ky tu \\r de cap nhat TAI CHO, nhung
# log tinh cua Kaggle commit KHONG hien thi kieu do (chi bat dong xuong dong that
# su) -> nhin log se thay "dung im" du van dang chay binh thuong. Vi vay IN THEM
# dong moi (print thuong, co flush=True) moi PROGRESS_EVERY anh, kem toc do va
# ETA, de theo doi duoc tien do khi xem log cua Save & Run All.
PROGRESS_EVERY = 50

t0=time.time()
n_err=0
n_todo = sum(1 for (q,fn,_) in worklist if (q,fn) not in done)
n_done_session = 0
print(f"Bat dau: {len(done)} mau (anh+cau hoi) da xong tu truoc, con {n_todo} mau can chay.", flush=True)
for i,(q, fn, gt) in enumerate(tqdm(worklist, desc="Zero-shot full-test", ncols=90)):
    if (q, fn) in done: continue
    try:
        raw = generate_answer(imgmap[fn], q_text[q])
    except Exception as e:
        # QUAN TRONG: khong de 1 anh loi/hong lam SAP toan bo vong lap (3891 mau ->
        # rui ro gap anh loi cao hon nhieu so voi mau 150/cau truoc day). Ghi nhan
        # loi, dung rong (se bi parse thanh nhan mac dinh), tiep tuc cac mau khac.
        n_err += 1
        print(f"[LOI anh {fn} / {q}]: {type(e).__name__}: {e}", flush=True)
        raw = ""
    pred = " và ".join(parse_multi_atoms(raw, q)) if q in WHAT_QIDS else parse_single_atom(raw, q)
    rows.append({"qid":q,"filename":fn,"raw_answer":raw,"pred_atoms":pred,"gt_answer":gt})
    n_done_session += 1
    if n_done_session % PROGRESS_EVERY == 0:
        elapsed = time.time()-t0
        rate = n_done_session/elapsed  # mau (anh+cau hoi) / giay, CUA PHIEN NAY
        remain = n_todo - n_done_session
        eta_s = remain/rate if rate>0 else 0
        now = _dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Da xong {len(rows)}/{len(worklist)} tong "
              f"({n_done_session}/{n_todo} phien nay) | "
              f"{rate:.2f} mau/s | da chay {elapsed/60:.1f} phut | "
              f"con lai uoc tinh {eta_s/60:.1f} phut | loi: {n_err}", flush=True)
    if len(rows) % 200 == 0:
        pd.DataFrame(rows).to_csv(CKPT, index=False)
gen_df = pd.DataFrame(rows)
gen_df.to_csv(CKPT, index=False)
print(f"Tong: {len(gen_df)} du doan, {n_err} loi anh (da bo qua), {time.time()-t0:.0f}s", flush=True)"""),

("markdown", """## 5. Chấm điểm theo chuẩn per-question atomic"""),

("code", """from sklearn.metrics import f1_score, accuracy_score

def score(gen_df):
    wf=[]; sf=[]; se=[]
    for q in QID_ORDER:
        sub = gen_df[gen_df["qid"]==q]
        if len(sub)==0: continue
        cds=[vocab[i] for i in cand[q]]
        gt_atoms=[answer_atoms(q,a) for a in sub["gt_answer"]]
        if q in WHAT_QIDS:
            pr_atoms=[set(a.split(" và ")) if a else set() for a in sub["pred_atoms"]]
            Yt=np.array([[1 if c in g else 0 for c in cds] for g in gt_atoms])
            Yp=np.array([[1 if c in p else 0 for c in cds] for p in pr_atoms])
            wf.append(f1_score(Yt,Yp,average="macro",zero_division=0))
        else:
            gt=[list(g)[0] if g else cds[-1] for g in gt_atoms]
            pr=list(sub["pred_atoms"])
            sf.append(f1_score(gt,pr,average="macro",labels=cds,zero_division=0))
            se.append(accuracy_score(gt,pr))
    return {"what_F1macro":float(np.mean(wf)) if wf else 0.0,
            "single_F1macro":float(np.mean(sf)) if sf else 0.0,
            "single_EM":float(np.mean(se)) if se else 0.0}

res = score(gen_df)
print("KET QUA Qwen2.5-VL-3B zero-shot (TOAN BO test set):", {k:round(v,4) for k,v in res.items()})
pd.DataFrame([{"mô hình":"Qwen2.5-VL-3B-Instruct (zero-shot, full test)", **res}]).to_csv(
    "/kaggle/working/qwen_zeroshot_fulltest_results.csv", index=False)

# recall theo tung nhan (de doi chieu voi ban fine-tune / Two-stage)
rec_rows=[]
for q in QID_ORDER:
    sub=gen_df[gen_df["qid"]==q]
    if len(sub)==0: continue
    tot=Counter(); hit=Counter()
    for _,x in sub.iterrows():
        gt=answer_atoms(q,x["gt_answer"]); pr=set(str(x["pred_atoms"]).split(" và ")) if pd.notna(x["pred_atoms"]) else set()
        for a in gt:
            tot[a]+=1
            if a in pr: hit[a]+=1
    for a in tot: rec_rows.append({"qid":q,"label":a,"n":tot[a],"recall":round(hit[a]/tot[a],4)})
pd.DataFrame(rec_rows).to_csv("/kaggle/working/qwen_zeroshot_fulltest_per_class_recall.csv", index=False)
print("Da luu qwen_zeroshot_fulltest_raw.csv, qwen_zeroshot_fulltest_results.csv, "
      "qwen_zeroshot_fulltest_per_class_recall.csv")"""),

("markdown", """## 6. Đóng gói kết quả để tải về"""),

("code", """import glob as _glob, zipfile
from IPython.display import FileLink
OUT="/kaggle/working/qwen_zeroshot_fulltest_ket_qua.zip"
if os.path.exists(OUT): os.remove(OUT)
with zipfile.ZipFile(OUT,"w",zipfile.ZIP_DEFLATED) as zf:
    for f in _glob.glob("/kaggle/working/qwen_zeroshot_fulltest_*.csv"):
        zf.write(f, os.path.basename(f)); print("+",os.path.basename(f))
print(f"\\nDa tao {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
FileLink("qwen_zeroshot_fulltest_ket_qua.zip")"""),
]
make_notebook(cells, os.path.join(HERE, "nb_qwen_zeroshot_fulltest.ipynb"))

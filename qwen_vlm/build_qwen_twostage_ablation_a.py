# -*- coding: utf-8 -*-
"""Sinh nb_qwen_twostage_ablation_a.ipynb — ABLATION Two-stage cRT, PHAN A (session
rieng, ~5h, an toan duoi tran 12h/session cua Kaggle):
  - r2_ep1 : RATIO_CAP=2.0, 1 epoch  -> day MANH lop thieu so hon (giu it lop da so)
  - r4_ep1 : RATIO_CAP=4.0, 1 epoch  -> giu NHIEU lop da so hon (giam mat mat lop da so)
Phan B (r3_ep2, 2 epoch) o file rieng: build_qwen_twostage_ablation_b.py.
Recipe train GIU NGUYEN 100% so voi Two-stage da chay on dinh — chi doi RATIO_CAP."""
import os
from _mknb import make_notebook
HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPT = r'''# ============================================================
# Qwen2.5-VL-3B + LoRA — GIAI DOAN 2 (cRT) — ABLATION (doc RATIO_CAP/EPOCH tu env)
# CHAY: RUN_TAG=.. RATIO_CAP=.. STAGE2_EPOCHS=.. torchrun --nproc_per_node=<N> qwen_train_stage2_abl.py
# ============================================================
import os, re, glob, sys, time
import numpy as np, torch, pandas as pd
from PIL import Image
from collections import Counter, defaultdict
from torch.utils.data import Dataset
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
    BitsAndBytesConfig, TrainingArguments, Trainer)
from qwen_vl_utils import process_vision_info
from peft import PeftModel, prepare_model_for_kbit_training
import warnings; warnings.filterwarnings("ignore")

LOCAL_RANK=int(os.environ.get("LOCAL_RANK",0)); RANK=int(os.environ.get("RANK",0))
WORLD_SIZE=int(os.environ.get("WORLD_SIZE",1))
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK); _=torch.zeros(1,device=f"cuda:{LOCAL_RANK}")
def rprint(*a,**k):
    if RANK==0: print(*a,**k)

# ── Cau hinh ablation doc tu ENV ──
RUN_TAG       = os.environ.get("RUN_TAG", "r3_ep1")
RATIO_CAP     = float(os.environ.get("RATIO_CAP", "3.0"))
STAGE2_EPOCHS = int(os.environ.get("STAGE2_EPOCHS", "1"))
rprint(f"\n########## ABLATION RUN: {RUN_TAG}  (RATIO_CAP={RATIO_CAP}, EPOCHS={STAGE2_EPOCHS}) ##########")

csv_train=glob.glob("/kaggle/input/**/train.csv", recursive=True)
imgs_any=(glob.glob("/kaggle/input/**/*.jpg", recursive=True)+
          glob.glob("/kaggle/input/**/*.png", recursive=True)+
          glob.glob("/kaggle/input/**/*.jpeg", recursive=True))
QA_DIR=os.path.dirname(csv_train[0])
IMG_DIR=Counter(os.path.dirname(p) for p in imgs_any).most_common(1)[0][0]
imgmap={os.path.basename(p):p for p in glob.glob(os.path.join(IMG_DIR,"*"))}

adapter_hits = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
assert adapter_hits, ("KHONG tim thay adapter_config.json trong /kaggle/input — "
    "ban can upload thu muc qwen_lora (adapter baseline da tai ve) thanh Kaggle "
    "Dataset moi va Add vao notebook nay truoc khi chay.")
STAGE1_DIR = os.path.dirname(adapter_hits[0])
rprint("Adapter giai doan 1 (baseline) tai:", STAGE1_DIR)

MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
OUT_DIR=f"/kaggle/working/qwen_lora_ts_{RUN_TAG}"
OUT_CSV=f"/kaggle/working/qwen_ablation_{RUN_TAG}_raw.csv"
RECALL_CSV=f"/kaggle/working/qwen_ablation_{RUN_TAG}_per_class_recall.csv"
METRIC_CSV=f"/kaggle/working/qwen_ablation_{RUN_TAG}_metrics.csv"

MIN_PIXELS=64*28*28; MAX_PIXELS=256*28*28
BATCH_SIZE=4; GRAD_ACCUM=1
STAGE2_LR=3e-5          # GIU NGUYEN nhu Two-stage on dinh: LR thap -> chi "chinh" khong "hoc lai"

SYSTEM_PROMPT=("Bạn là trợ lý phân tích ảnh vi phạm sử dụng vỉa hè. "
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
vocab,idx,cand=build_vocab(pd.concat([train_df,val_df,test_df],ignore_index=True))
q_text={q: pd.concat([train_df,val_df,test_df])[lambda d:d["qid"]==q]["question"].mode().iloc[0] for q in QID_ORDER}

def parse_multi_atoms(raw_text,qid):
    t=normalize(raw_text); cds=sorted((vocab[i] for i in cand[qid]),key=len,reverse=True)
    found=[]
    for c in cds:
        if c in t and not any(c in f for f in found): found.append(c)
    return found if found else [cds[-1]]
def parse_single_atom(raw_text,qid):
    t=normalize(raw_text); cds=sorted((vocab[i] for i in cand[qid]),key=len,reverse=True)
    found=[]
    for c in cds:
        if c in t and not any(c in f for f in found): found.append(c)
    if not found: return cds[-1]
    neg=[f for f in found if f.startswith("không")]
    return neg[0] if neg else found[0]

def build_messages(image,question,answer_text=None):
    msgs=[{"role":"system","content":SYSTEM_PROMPT},
          {"role":"user","content":[{"type":"image","image":image},{"type":"text","text":question}]}]
    if answer_text is not None: msgs.append({"role":"assistant","content":answer_text})
    return msgs

# ── PHUONG PHAP: xay tap GIAM MAU lop da so (RUS), giu nguyen lop hiem ──
def build_stage2_subset(df, ratio_cap, seed=42):
    rng = np.random.default_rng(seed)
    freq = defaultdict(Counter)
    for _, r in df.iterrows():
        q = str(r["qid"])
        for a in answer_atoms(q, r["answer"]): freq[q][a] += 1
    qmin = {q: min(c.values()) for q, c in freq.items()}
    per_group_rows = defaultdict(list)
    keep_idx = []
    for i, r in df.iterrows():
        q = str(r["qid"]); atoms = answer_atoms(q, r["answer"])
        if not atoms: keep_idx.append(i); continue
        rarest = min(atoms, key=lambda a: freq[q][a])
        per_group_rows[(q, rarest)].append(i)
    for (q, lab), idxs in per_group_rows.items():
        cap = int(qmin[q] * ratio_cap)
        if len(idxs) > cap:
            idxs = list(rng.choice(idxs, size=cap, replace=False))
        keep_idx.extend(idxs)
    return df.loc[sorted(set(keep_idx))].reset_index(drop=True)

n_before = len(train_df)
stage2_df = build_stage2_subset(train_df, RATIO_CAP)
if RANK == 0:
    print(f"[{RUN_TAG}] Stage-2 subset: {n_before} -> {len(stage2_df)} dong "
          f"(con {100*len(stage2_df)/n_before:.0f}%, undersample lop da so ve toi da {RATIO_CAP}x lop hiem nhat)")

class VQASFTDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True); self.processor = processor
        self.image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]; p = imgmap.get(r["filename"])
        try: image = Image.open(p).convert("RGB") if p else Image.new("RGB",(336,336))
        except Exception: image = Image.new("RGB",(336,336))
        question = q_text[str(r["qid"])]; answer = str(r["answer"]).strip()
        msgs_full = build_messages(image, question, answer)
        msgs_prompt = build_messages(image, question, None)
        text_full = self.processor.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)
        text_prompt = self.processor.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)
        image_inputs,_ = process_vision_info(msgs_full)
        enc = self.processor(text=[text_full], images=image_inputs, padding=False, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        num_img = int((input_ids == self.image_pad_id).sum())
        prompt_ids = self.processor.tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids) - 1 + num_img
        labels = input_ids.clone(); labels[:prompt_len] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None: labels[labels == pad_id] = -100
        return {"input_ids": input_ids, "attention_mask": enc["attention_mask"].squeeze(0),
                "pixel_values": enc["pixel_values"], "image_grid_thw": enc["image_grid_thw"], "labels": labels}

def make_collate_fn(processor):
    pad_id = processor.tokenizer.pad_token_id
    def collate_fn(batch):
        maxlen = max(b["input_ids"].shape[-1] for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            n = b["input_ids"].shape[-1]; pd_ = maxlen - n
            input_ids.append(torch.cat([b["input_ids"], torch.full((pd_,), pad_id, dtype=b["input_ids"].dtype)]))
            attn.append(torch.cat([b["attention_mask"], torch.zeros(pd_, dtype=b["attention_mask"].dtype)]))
            labels.append(torch.cat([b["labels"], torch.full((pd_,), -100, dtype=b["labels"].dtype)]))
        return {"input_ids": torch.stack(input_ids), "attention_mask": torch.stack(attn),
                "labels": torch.stack(labels),
                "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
                "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch], dim=0)}
    return collate_fn

rprint(f"[DDP] WORLD_SIZE={WORLD_SIZE} — nap base model + adapter GIAI DOAN 1 (is_trainable=True)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
processor.tokenizer.padding_side = "right"
if processor.tokenizer.pad_token_id is None: processor.tokenizer.pad_token = processor.tokenizer.eos_token
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=bnb, torch_dtype=torch.float16,
    device_map={"": f"cuda:{LOCAL_RANK}"}, low_cpu_mem_usage=True)
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False})
model = PeftModel.from_pretrained(base_model, STAGE1_DIR, is_trainable=True)
if RANK==0: model.print_trainable_parameters()

train_ds = VQASFTDataset(stage2_df, processor)
rprint(f"[{RUN_TAG}] Stage-2 train: {len(train_ds)} dong, LR={STAGE2_LR}, epochs={STAGE2_EPOCHS}")

training_args = TrainingArguments(
    output_dir=OUT_DIR, num_train_epochs=STAGE2_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=STAGE2_LR, lr_scheduler_type="cosine", warmup_ratio=0.03, fp16=True,
    logging_steps=50, save_strategy="epoch", save_total_limit=1,
    dataloader_num_workers=2, dataloader_pin_memory=True,
    remove_unused_columns=False, report_to="none", label_names=["labels"],
    ddp_find_unused_parameters=False)
trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                  data_collator=make_collate_fn(processor))
rprint(f"[{RUN_TAG}] Bat dau fine-tune GIAI DOAN 2...")
trainer.train()
rprint(f"[{RUN_TAG}] Giai doan 2 xong.")
if RANK == 0:
    model.save_pretrained(OUT_DIR); processor.save_pretrained(OUT_DIR)
    rprint(f"Adapter luu -> {OUT_DIR}")
if WORLD_SIZE > 1:
    torch.distributed.barrier(); torch.distributed.destroy_process_group()
if RANK != 0: sys.exit(0)

print(f"\n[{RUN_TAG}] Danh gia tren TOAN BO test set...")
model = trainer.model; model.config.use_cache = True
model.gradient_checkpointing_disable(); model.eval()
processor.tokenizer.padding_side = "left"
for _k in ("temperature","top_p","top_k"):
    if hasattr(model.generation_config, _k): setattr(model.generation_config, _k, None)
model.generation_config.do_sample = False

eval_rows=[]
for q in QID_ORDER:
    sub = test_df[test_df["qid"]==q]
    for _, r in sub.iterrows():
        if imgmap.get(r["filename"]) is not None:
            eval_rows.append({"qid":q,"filename":r["filename"],"gt_answer":r["answer"]})
eval_df = pd.DataFrame(eval_rows)
rprint(f"Tong mau eval (toan bo test): {len(eval_df)}")

from tqdm.auto import tqdm
@torch.no_grad()
def generate_ft(image_path, question):
    msgs = build_messages(image_path, question, None)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs,_ = process_vision_info(msgs)
    enc = processor(text=[text], images=imgs, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=24, do_sample=False)
    gen = out[:, enc["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

rows=[]; t0=time.time()
for _, r in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"Eval {RUN_TAG}", ncols=90):
    p = imgmap.get(r["filename"])
    if p is None: continue
    raw = generate_ft(p, q_text[r["qid"]])
    pred = " và ".join(parse_multi_atoms(raw, r["qid"])) if r["qid"] in WHAT_QIDS else parse_single_atom(raw, r["qid"])
    rows.append({"qid":r["qid"],"filename":r["filename"],"raw_answer":raw,"pred_atoms":pred,"gt_answer":r["gt_answer"]})
pred_df = pd.DataFrame(rows); pred_df.to_csv(OUT_CSV, index=False)

from sklearn.metrics import f1_score, accuracy_score
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

wF,sF,sEM = score(pred_df)
pd.DataFrame([{"run_tag":RUN_TAG,"ratio_cap":RATIO_CAP,"epochs":STAGE2_EPOCHS,
    "what_F1macro":round(wF,4),"single_F1macro":round(sF,4),"single_EM":round(sEM,4)}]).to_csv(METRIC_CSV, index=False)

rec_rows=[]
for q in QID_ORDER:
    sub=pred_df[pred_df["qid"]==q]
    if len(sub)==0: continue
    tot=Counter(); hit=Counter()
    for _,x in sub.iterrows():
        gt=answer_atoms(q,x["gt_answer"]); pr=set(str(x["pred_atoms"]).split(" và ")) if pd.notna(x["pred_atoms"]) else set()
        for a in gt:
            tot[a]+=1
            if a in pr: hit[a]+=1
    for a in tot: rec_rows.append({"qid":q,"label":a,"n":tot[a],"recall":round(hit[a]/tot[a],4)})
pd.DataFrame(rec_rows).to_csv(RECALL_CSV, index=False)

print(f"\n=== [{RUN_TAG}] KET QUA (RATIO_CAP={RATIO_CAP}, EPOCHS={STAGE2_EPOCHS}, LR2={STAGE2_LR}) ===")
print(f"what F1-macro={wF:.4f}  single F1-macro={sF:.4f}  single EM={sEM:.4f}  ({time.time()-t0:.0f}s)")
watch={"Q9":"có","Q10":"không","Q2":"sửa xe/rửa xe","Q8":"bục/bậc tự xây"}
for q,lab in watch.items():
    m=[r for r in rec_rows if r["qid"]==q and r["label"]==lab]
    if m: print(f"  {q} lop {lab!r}: recall={m[0]['recall']:.3f} (n={m[0]['n']})")
print(f"Da luu: {OUT_CSV}, {RECALL_CSV}, {METRIC_CSV}", flush=True)
'''

cells = [
("markdown", """# Ablation Two-stage cRT — PHẦN A: RATIO_CAP 2.0 và 4.0 (~5h)

Two-stage cRT là phương pháp **duy nhất chạy ổn định qua nhiều lần** (3 phương pháp
kia — Oversampling / Label-Smoothing / Weighted-loss — đều sập NaN). Notebook này
**không thử phương pháp mới**, mà **quét tham số** để tìm điểm vận hành tốt nhất.

**2 cấu hình chạy tuần tự trong notebook** (mỗi cấu hình là 1 lần `torchrun` RIÊNG
→ tiến trình sạch, một lần lỗi không kéo theo cái kia):

| Run tag | RATIO_CAP | Epoch | Ý nghĩa |
|---|---|---|---|
| `r2_ep1` | 2.0 | 1 | Undersample MẠNH hơn → đẩy lớp thiểu số nhiều hơn (nhưng bỏ nhiều lớp đa số) |
| `r4_ep1` | 4.0 | 1 | Giữ NHIỀU lớp đa số hơn → giảm mức sụt của lớp đa số |

> Bản đã chạy trước là `RATIO_CAP=3.0, 1 epoch` (what=0.868 / single=0.892 / EM=0.935).
> Phần A này bổ sung 2 điểm quanh nó. **Phần B** (`r3_ep2` — cùng ratio 3.0 nhưng 2
> epoch) nằm ở **notebook riêng**: `nb_qwen_twostage_ablation_b.ipynb`.

**Recipe train giữ NGUYÊN 100%** so với bản Two-stage đã ổn định (cùng LR=3e-5,
cùng batch, cùng grad-checkpoint) — chỉ thay `RATIO_CAP`. Vì vậy **rủi ro sập gần
như bằng 0** (cùng họ recipe undersampling + CE chuẩn đã chứng minh an toàn).

## ⚠️ Cần làm trước khi chạy: upload adapter baseline
1. Kaggle → **Datasets → New Dataset** → upload thư mục `qwen_lora` (adapter baseline
   giai đoạn 1) → Create.
2. **Add Input** dataset đó vào notebook này (cùng dataset ảnh+CSV gốc).
3. Notebook tự dò `adapter_config.json` trong `/kaggle/input/**`.

> ⚠️ Bật **Internet** + **GPU T4 ×2**. Mỗi cấu hình ~2.5h → tổng **~5h, an toàn
> tuyệt đối dưới trần 12h/session của Kaggle**."""),

("code", """# ===== 0. Cai thu vien =====
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","-U",
    "transformers>=4.49.0","accelerate>=0.34.0","qwen-vl-utils",
    "bitsandbytes>=0.46.1","peft","tqdm"], check=False)
print("Da cai xong thu vien.")"""),

("code", "%%writefile qwen_train_stage2_abl.py\n" + SCRIPT),

("markdown", """## 2. Chạy 2 cấu hình (mỗi cấu hình = 1 torchrun riêng)"""),

("code", """import subprocess as _sp, time as _t2, os as _os, torch

CONFIGS = [
    {"RUN_TAG":"r2_ep1", "RATIO_CAP":"2.0", "STAGE2_EPOCHS":"1"},
    {"RUN_TAG":"r4_ep1", "RATIO_CAP":"4.0", "STAGE2_EPOCHS":"1"},
]

def clean_gpu():
    try:
        q=_sp.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],capture_output=True,text=True)
        mypid=_os.getpid(); stray=sorted({int(x) for x in q.stdout.split() if x.strip().isdigit()}-{mypid})
        for pid in stray:
            try: _os.kill(pid,9); print("Da kill GPU pid mo coi:",pid)
            except Exception: pass
        if stray: _t2.sleep(4)
    except Exception as e:
        print("(bo qua don:",e,")")

N_GPU=torch.cuda.device_count(); n=max(N_GPU,1)
for cfg in CONFIGS:
    clean_gpu()
    _env={**_os.environ,"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True", **cfg}
    print(f"\\n===== CHAY {cfg['RUN_TAG']}: torchrun --nproc_per_node={n} =====", flush=True)
    ret=_sp.run(["torchrun",f"--nproc_per_node={n}","qwen_train_stage2_abl.py"], env=_env)
    print(f"[{cfg['RUN_TAG']}] Torchrun exit code:", ret.returncode, flush=True)
print("\\nXONG CA 2 CAU HINH (PHAN A).")"""),

("markdown", """## 3. Tổng hợp bảng ablation (baseline vs 3.0 cũ vs 2.0/4.0 mới)"""),

("code", """import pandas as pd, glob
BASE = {"run_tag":"baseline (mẫu 150/câu)","ratio_cap":"-","epochs":"-",
        "what_F1macro":0.831,"single_F1macro":0.867,"single_EM":0.933}
TS3  = {"run_tag":"r3_ep1 (đã chạy)","ratio_cap":3.0,"epochs":1,
        "what_F1macro":0.868,"single_F1macro":0.892,"single_EM":0.935}
rows=[BASE, TS3]
for f in sorted(glob.glob("/kaggle/working/qwen_ablation_*_metrics.csv")):
    rows.append(pd.read_csv(f).iloc[0].to_dict())
tab = pd.DataFrame(rows)
print(tab.to_string(index=False))
tab.to_csv("/kaggle/working/qwen_ablation_a_summary.csv", index=False)

print("\\n=== RECALL LOP THIEU SO theo tung cau hinh ===")
watch=[("Q9","có"),("Q10","không"),("Q2","sửa xe/rửa xe"),("Q8","bục/bậc tự xây"),
       ("Q4","không"),("Q9","không")]  # 2 dong cuoi = lop DA SO de theo doi mat mat
for f in sorted(glob.glob("/kaggle/working/qwen_ablation_*_per_class_recall.csv")):
    tag=f.split("qwen_ablation_")[1].split("_per_class")[0]
    rec=pd.read_csv(f)
    print(f"\\n[{tag}]")
    for q,lab in watch:
        m=rec[(rec["qid"]==q)&(rec["label"]==lab)]
        if len(m): print(f"  {q:4} {lab:20} recall={m.iloc[0]['recall']:.3f} (n={m.iloc[0]['n']})")"""),

("markdown", """## 4. Đóng gói kết quả + adapter để tải về"""),

("code", """import os, glob, zipfile
from IPython.display import FileLink
OUT="/kaggle/working/qwen_ablation_a_ket_qua.zip"
if os.path.exists(OUT): os.remove(OUT)
with zipfile.ZipFile(OUT,"w",zipfile.ZIP_DEFLATED) as zf:
    for f in glob.glob("/kaggle/working/qwen_ablation_*.csv"):
        zf.write(f, os.path.basename(f)); print("+",os.path.basename(f))
    zf.write("qwen_train_stage2_abl.py","qwen_train_stage2_abl.py")
    for ad in glob.glob("/kaggle/working/qwen_lora_ts_*"):
        if os.path.isdir(ad):
            for root,_,files in os.walk(ad):
                for name in files:
                    full=os.path.join(root,name); zf.write(full, os.path.relpath(full,"/kaggle/working"))
            print("+", os.path.basename(ad),"(adapter)")
print(f"\\nDa tao {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
FileLink("qwen_ablation_a_ket_qua.zip")"""),
]
make_notebook(cells, os.path.join(HERE, "nb_qwen_twostage_ablation_a.ipynb"))

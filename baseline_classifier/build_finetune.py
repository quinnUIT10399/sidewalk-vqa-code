# -*- coding: utf-8 -*-
"""Sinh nb_finetune_kaggle.ipynb — FINE-TUNE encoder anh (CLIP ViT-L/14) tren
Kaggle GPU. Self-contained (khong import module local). So sanh:
  (A) ViT-L/14 DONG BANG + MLP head  (linear-probe, tai lap ket qua local)
  (B) ViT-L/14 FINE-TUNE (mo khoa vai block cuoi) + head, end-to-end
  (C) (B) + Focal loss
Cham diem theo DUNG chuan per-question atomic (giong unified.py o local)."""
import os
from _mknb import make_notebook
HERE = os.path.dirname(os.path.abspath(__file__))

cells = [
("markdown", """# Fine-tune CLIP ViT-L/14 cho Sidewalk-VQA (Kaggle GPU)

Mô hình **dùng chung** (một model / 10 câu), câu hỏi mã hoá bằng **PhoBERT (đóng
băng)**, ảnh mã hoá bằng **CLIP ViT-L/14** — encoder đã chứng minh tốt hơn ViT-B/32
ở mọi chỉ số trong thực nghiệm linear-probe (local, CPU). Trên Kaggle có GPU nên
fine-tune ViT-L/14 (thay vì B/32) là khả thi và cho mốc mạnh nhất.

Thí nghiệm so sánh 3 cấu hình để trả lời "fine-tune encoder có hơn linear-probe
không, và phương pháp cân bằng còn tác dụng không khi model mạnh hơn":

- **(A) Frozen** — ViT-L/14 đóng băng + MLP head (linear-probe, tái lập kết quả local).
- **(B) Fine-tune** — mở khoá 2 block cuối của ViT-L/14 + head, huấn luyện end-to-end
  (ViT-L nặng hơn B/32 ~4 lần nên mở khoá ít block hơn để vừa VRAM T4 16GB).
- **(C) Fine-tune + Focal** — như (B) nhưng đổi loss sang Focal (phương pháp cân bằng).

Chấm điểm **cùng chuẩn per-question atomic** như ở local (câu *what* = đa nhãn nguyên
tử; câu còn lại = đơn nhãn), nên số liệu xếp thẳng vào bảng chung được.

> ⚠️ **BẬT INTERNET** trong Kaggle (Settings → Internet → On) để tải trọng số
> ViT-L/14 (timm) và PhoBERT (HuggingFace). Chọn **GPU T4** (Settings → Accelerator).
> Nếu gặp lỗi hết VRAM (CUDA OOM): giảm `bs` (batch size) trong `train_model(...)`
> xuống 16 hoặc giảm `n_unfreeze` xuống 1.

> Bài học LLaVA: ta KHÔNG dùng VLM sinh; PhoBERT chỉ mã hoá 10 câu hỏi ngắn cố định
> (cắt max_length=64) nên không thể tràn token."""),

("code", """# ===== 0. Cau hinh & do dac duong dan Kaggle =====
import os, re, glob, time, math, random
import numpy as np, pandas as pd
import torch, torch.nn as nn
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, "| GPU:", torch.cuda.get_device_name(0) if DEVICE=="cuda" else "—")

# Tu dong tim thu muc anh + cac file csv trong /kaggle/input/**
def find_dir_with(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    return hits
csv_train = find_dir_with("train.csv")
imgs_any  = glob.glob("/kaggle/input/**/*.jpg", recursive=True) + glob.glob("/kaggle/input/**/*.png", recursive=True) + glob.glob("/kaggle/input/**/*.jpeg", recursive=True)
assert csv_train, "Khong tim thay train.csv trong /kaggle/input — kiem tra lai dataset da add."
assert imgs_any,  "Khong tim thay anh (.jpg/.png) trong /kaggle/input — kiem tra lai dataset anh."
QA_DIR  = os.path.dirname(csv_train[0])
# thu muc anh = thu muc chua nhieu anh nhat
from collections import Counter
IMG_DIR = Counter(os.path.dirname(p) for p in imgs_any).most_common(1)[0][0]
print("QA_DIR :", QA_DIR)
print("IMG_DIR:", IMG_DIR, "| so anh:", sum(1 for p in imgs_any if os.path.dirname(p)==IMG_DIR))"""),

("code", """# ===== 1. Nap annotation + xay GLOBAL ATOM VOCAB (giong unified.py local) =====
train = pd.read_csv(os.path.join(QA_DIR,"train.csv")); train["qid"]=train["qid"].astype(str)
val   = pd.read_csv(os.path.join(QA_DIR,"val.csv"));   val["qid"]=val["qid"].astype(str)
test  = pd.read_csv(os.path.join(QA_DIR,"test.csv"));  test["qid"]=test["qid"].astype(str)

QID_ORDER=["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10"]
WHAT_QIDS={"Q2","Q5","Q8"}; DERIVED_BIN={"Q4"}
SINGLE_QIDS=[q for q in QID_ORDER if q not in WHAT_QIDS]
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
A=len(vocab)
print(f"GLOBAL atom vocab ({A}):", vocab)"""),

("code", """# ===== 2. PhoBERT (DONG BANG): ma hoa 10 cau hoi co dinh mot lan =====
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("vinai/phobert-base")
pho = AutoModel.from_pretrained("vinai/phobert-base").to(DEVICE).eval()
q_text = {q: pd.concat([train,val,test])[lambda d:d["qid"]==q]["question"].mode().iloc[0] for q in QID_ORDER}
@torch.no_grad()
def encode_q(t):
    enc=tok(t,return_tensors="pt",truncation=True,max_length=64,padding="max_length").to(DEVICE)
    out=pho(**enc); mask=enc["attention_mask"].unsqueeze(-1).float()
    return ((out.last_hidden_state*mask).sum(1)/mask.sum(1).clamp(min=1e-9)).squeeze(0).cpu().numpy()
QVEC = {q: encode_q(q_text[q]).astype("float32") for q in QID_ORDER}
TEXT_DIM = next(iter(QVEC.values())).shape[0]
del pho; torch.cuda.empty_cache()
print("PhoBERT question dim:", TEXT_DIM, "(da cache 10 vector, giai phong PhoBERT khoi GPU)")"""),

("code", """# ===== 3. Dataset anh + transform CLIP =====
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
VIT_NAME="vit_large_patch14_clip_224.openai"   # 1024-d, encoder tot nhat da xac nhan o local
_tmp=timm.create_model(VIT_NAME,pretrained=True,num_classes=0)
cfg=timm.data.resolve_data_config({},model=_tmp); IMG_DIM=_tmp.num_features; del _tmp
tf_train=transforms.Compose([
    transforms.Resize(cfg["input_size"][1]+16), transforms.RandomCrop(cfg["input_size"][1]),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize(cfg["mean"],cfg["std"])])
tf_eval=transforms.Compose([
    transforms.Resize(cfg["input_size"][1]), transforms.CenterCrop(cfg["input_size"][1]),
    transforms.ToTensor(), transforms.Normalize(cfg["mean"],cfg["std"])])
IMGPATH={os.path.basename(p):p for p in glob.glob(os.path.join(IMG_DIR,"*"))}

class VQADS(Dataset):
    def __init__(self, df, train=False):
        self.rows=[r for _,r in df.iterrows() if r["filename"] in IMGPATH]
        self.tf=tf_train if train else tf_eval
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; q=str(r["qid"])
        try: img=Image.open(IMGPATH[r["filename"]]).convert("RGB")
        except Exception: img=Image.new("RGB",(224,224))
        y=np.zeros(A,dtype="float32")
        for a in answer_atoms(q,r["answer"]):
            if a in idx: y[idx[a]]=1.0
        return self.tf(img), torch.tensor(QVEC[q]), torch.tensor(y), q, r["filename"]
print("IMG_DIM (ViT-L/14):", IMG_DIM, "| n_train rows co anh:", len(VQADS(train)))"""),

("code", """# ===== 4. Ham cham diem theo chuan per-question atomic =====
from sklearn.metrics import f1_score, accuracy_score, recall_score
from collections import Counter as _Counter
def decode_and_score(P, ys, qs, filenames=None):
    # P: (N,A) xac suat sigmoid; ys: (N,A) multihot that; qs: list qid
    qs=np.array(qs); preds=np.zeros_like(P,dtype=int)
    for i in range(len(P)):
        cd=cand[qs[i]]; sc=P[i,cd]
        if qs[i] in WHAT_QIDS:
            take=[c for c,s in zip(cd,sc) if s>=0.5]
            if not take: take=[cd[int(np.argmax(sc))]]
            preds[i,take]=1
        else:
            preds[i,cd[int(np.argmax(sc))]]=1
    # what: F1-macro nguyen tu tung cau roi trung binh (bo qua cau khong co mau trong batch/test)
    wf=[]
    for q in WHAT_QIDS:
        m=qs==q
        if m.sum()==0: continue
        cd=cand[q]
        wf.append(f1_score(ys[m][:,cd],preds[m][:,cd],average="macro",zero_division=0))
    # single: F1-macro tung cau (khong gian nhan rieng) roi trung binh
    sf=[]; se=[]
    for q in SINGLE_QIDS:
        m=qs==q
        if m.sum()==0: continue
        cd=cand[q]; classes=[vocab[c] for c in cd]
        gt=[classes[int(np.argmax(t))] for t in ys[m][:,cd]]
        pr=[classes[int(np.argmax(p))] for p in preds[m][:,cd]]
        sf.append(f1_score(gt,pr,average="macro",labels=classes,zero_division=0))
        se.append(accuracy_score(gt,pr))
    metrics = {"what_F1macro":float(np.mean(wf)) if wf else 0.0,
               "single_F1macro":float(np.mean(sf)) if sf else 0.0,
               "single_EM":float(np.mean(se)) if se else 0.0}
    if filenames is None:
        return metrics
    # ===== xuat du doan TUNG MAU (de phan tich loi) + recall theo tung nhan =====
    raw_rows=[]; rec_tot=_Counter(); rec_hit=_Counter()
    for i in range(len(P)):
        q=qs[i]
        gt_labels=[vocab[c] for c in cand[q] if ys[i,c]==1]
        pr_labels=[vocab[c] for c in cand[q] if preds[i,c]==1]
        raw_rows.append({"qid":q,"filename":filenames[i],
                          "gt_answer":" và ".join(gt_labels),
                          "pred_answer":" và ".join(pr_labels)})
        for lab in gt_labels:
            rec_tot[(q,lab)]+=1
            if lab in pr_labels: rec_hit[(q,lab)]+=1
    raw_df=pd.DataFrame(raw_rows)
    rec_df=pd.DataFrame([{"qid":q,"label":lab,"n":n,"recall":round(rec_hit[(q,lab)]/n,4)}
                         for (q,lab),n in rec_tot.items()])
    return metrics, raw_df, rec_df"""),

("code", """# ===== 5. Model: ViT-L/14 + fusion head =====
class VQAModel(nn.Module):
    def __init__(self, n_unfreeze=0, hidden=512, p=0.3):
        super().__init__()
        self.vit=timm.create_model(VIT_NAME,pretrained=True,num_classes=0)
        for prm in self.vit.parameters(): prm.requires_grad=False
        if n_unfreeze>0:
            for blk in self.vit.blocks[-n_unfreeze:]:
                for prm in blk.parameters(): prm.requires_grad=True
            for prm in self.vit.norm.parameters(): prm.requires_grad=True
        self.head=nn.Sequential(nn.Linear(TEXT_DIM+IMG_DIM,hidden),nn.ReLU(),
                                nn.Dropout(p),nn.Linear(hidden,A))
    def forward(self,pixel,qvec):
        img=self.vit(pixel); return self.head(torch.cat([qvec,img],1))

def sigmoid_focal(logits,target,gamma=2.0):
    p=torch.sigmoid(logits)
    ce=nn.functional.binary_cross_entropy_with_logits(logits,target,reduction="none")
    pt=p*target+(1-p)*(1-target); return ((1-pt)**gamma*ce).mean()

@torch.no_grad()
def evaluate(model, loader, return_raw=False):
    model.eval(); Ps=[]; Ys=[]; Qs=[]; Fs=[]
    for pixel,qv,y,q,fn in loader:
        pixel,qv=pixel.to(DEVICE),qv.to(DEVICE)
        with torch.autocast(device_type="cuda",enabled=DEVICE=="cuda"):
            logit=model(pixel,qv)
        Ps.append(torch.sigmoid(logit).float().cpu().numpy()); Ys.append(y.numpy()); Qs+=list(q); Fs+=list(fn)
    P=np.concatenate(Ps); Y=np.concatenate(Ys)
    if return_raw:
        return decode_and_score(P,Y,Qs,filenames=Fs)
    return decode_and_score(P,Y,Qs)"""),

("markdown", """### Trích đặc trưng ViT-L/14 MỘT LẦN (cho cấu hình frozen)

Vì (A) đóng băng hoàn toàn ViT, không cần forward lại ảnh qua ViT mỗi epoch (đó là
lý do (A) chạy chậm ở lần trước — 305s/epoch dù không cập nhật trọng số ViT). Ở đây
trích đặc trưng ảnh **một lần duy nhất** (giống cách làm ở local), lưu vào bộ nhớ,
rồi chỉ train MLP head trên đặc trưng đã cache — nhanh hơn hàng chục lần."""),

("code", """@torch.no_grad()
def extract_vit_features(df, bs=64):
    vit=timm.create_model(VIT_NAME,pretrained=True,num_classes=0).to(DEVICE).eval()
    ds=VQADS(df, train=False)  # tf_eval: khong augment, giong luc trich cache o local
    loader=DataLoader(ds,batch_size=bs,shuffle=False,num_workers=2,pin_memory=True)
    feats=[]; ys=[]; qs=[]; fns=[]
    for pixel,qv,y,q,fn in loader:
        pixel=pixel.to(DEVICE)
        with torch.autocast(device_type="cuda",enabled=DEVICE=="cuda"):
            f=vit(pixel)
        feats.append(f.float().cpu().numpy()); ys.append(y.numpy()); qs+=list(q); fns+=list(fn)
    del vit; torch.cuda.empty_cache()
    return np.concatenate(feats), np.concatenate(ys), np.array(qs), fns

def train_frozen_head(loss_name="bce", epochs=80, bs=256, lr=1e-3):
    t0=time.time()
    Ftr,Ytr,Qtr,_ = extract_vit_features(train)
    Fte,Yte,Qte,Fte_fn = extract_vit_features(test)
    print(f"  Da trich dac trung ViT-L/14 mot lan: train={Ftr.shape} test={Fte.shape} ({time.time()-t0:.0f}s)")
    Xtr=np.concatenate([np.stack([QVEC[q] for q in Qtr]), Ftr], axis=1).astype("float32")
    Xte=np.concatenate([np.stack([QVEC[q] for q in Qte]), Fte], axis=1).astype("float32")
    head=nn.Sequential(nn.Linear(TEXT_DIM+IMG_DIM,512),nn.ReLU(),nn.Dropout(0.3),nn.Linear(512,A)).to(DEVICE)
    opt=torch.optim.Adam(head.parameters(),lr=lr,weight_decay=1e-4)
    Xt=torch.tensor(Xtr,device=DEVICE); Yt=torch.tensor(Ytr,device=DEVICE); n=len(Xt)
    Xte_t=torch.tensor(Xte,device=DEVICE)
    for ep in range(epochs):
        head.train(); perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs]; opt.zero_grad()
            logit=head(Xt[b])
            loss=sigmoid_focal(logit,Yt[b]) if loss_name=="focal" else nn.functional.binary_cross_entropy_with_logits(logit,Yt[b])
            loss.backward(); opt.step()
        if (ep+1)%10==0 or ep==epochs-1:
            head.eval()
            with torch.no_grad(): P=torch.sigmoid(head(Xte_t)).cpu().numpy()
            m=decode_and_score(P,Yte,Qte)
            print(f"  ep{ep+1}/{epochs} whatF1={m['what_F1macro']:.4f} singleF1={m['single_F1macro']:.4f} EM={m['single_EM']:.4f}")
    head.eval()
    with torch.no_grad(): P=torch.sigmoid(head(Xte_t)).cpu().numpy()
    _, raw_df, rec_df = decode_and_score(P,Yte,Qte,filenames=Fte_fn)
    return head, m, raw_df, rec_df"""),

("code", """# ===== 6. Vong huan luyen chung (dung cho (B)/(C) — co fine-tune ViT, KHONG cache duoc) =====
def train_model(n_unfreeze, loss_name="bce", epochs=8, bs=32, lr_head=1e-3, lr_vit=1e-5):
    tr_loader=DataLoader(VQADS(train,train=True),batch_size=bs,shuffle=True,num_workers=2,pin_memory=True,drop_last=True)
    te_loader=DataLoader(VQADS(test,train=False),batch_size=128,shuffle=False,num_workers=2,pin_memory=True)
    model=VQAModel(n_unfreeze=n_unfreeze).to(DEVICE)
    vit_params=[p for p in model.vit.parameters() if p.requires_grad]
    groups=[{"params":model.head.parameters(),"lr":lr_head}]
    if vit_params: groups.append({"params":vit_params,"lr":lr_vit})
    opt=torch.optim.AdamW(groups,weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler(enabled=DEVICE=="cuda")
    m = None
    for ep in range(epochs):
        model.train(); t0=time.time(); tot=0;nb=0
        for pixel,qv,y,q,fn in tr_loader:
            pixel,qv,y=pixel.to(DEVICE),qv.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            with torch.autocast(device_type="cuda",enabled=DEVICE=="cuda"):
                logit=model(pixel,qv)
                loss=sigmoid_focal(logit,y) if loss_name=="focal" else nn.functional.binary_cross_entropy_with_logits(logit,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot+=loss.item(); nb+=1
        m=evaluate(model,te_loader)
        print(f"  ep{ep+1}/{epochs} loss={tot/nb:.4f} whatF1={m['what_F1macro']:.4f} singleF1={m['single_F1macro']:.4f} EM={m['single_EM']:.4f} ({time.time()-t0:.0f}s)")
    _, raw_df, rec_df = evaluate(model,te_loader,return_raw=True)
    return model, m, raw_df, rec_df"""),

("markdown", """## (A) Frozen — ViT-L/14 đóng băng + MLP head (linear-probe)

Trích đặc trưng ViT-L/14 MỘT LẦN (nhanh, chỉ forward, dùng GPU), sau đó chỉ train
MLP head trên đặc trưng đã cache (như ở local) — nên rất nhanh (giây, không phải
5+ phút/epoch như bản trước đó lỡ forward lại ViT mỗi epoch)."""),

("code", """print("(A) FROZEN linear-probe (ViT-L/14, dac trung cache 1 lan)"); torch.manual_seed(SEED)
modelA, resA, rawA, recA = train_frozen_head(loss_name="bce", epochs=80, bs=256, lr=1e-3)
print("KET QUA (A):", {k:round(v,4) for k,v in resA.items()})
del modelA; torch.cuda.empty_cache()"""),

("markdown", """## (B) Fine-tune — mở khoá 2 block cuối ViT-L/14, end-to-end

ViT-L/14 có 24 block (gấp đôi ViT-B/32) và nặng hơn nhiều — chỉ mở khoá 2 block
cuối (thay vì 4) để vừa VRAM T4 16GB và tránh overfit vì test set không lớn."""),

("code", """print("(B) FINE-TUNE 2 block cuoi (ViT-L/14)"); torch.manual_seed(SEED)
modelB, resB, rawB, recB = train_model(n_unfreeze=2, loss_name="bce", epochs=6, bs=32, lr_head=1e-3, lr_vit=1e-5)
print("KET QUA (B):", {k:round(v,4) for k,v in resB.items()})
del modelB; torch.cuda.empty_cache()"""),

("markdown", """## (C) Fine-tune + Focal (phương pháp cân bằng)"""),

("code", """print("(C) FINE-TUNE + Focal (ViT-L/14)"); torch.manual_seed(SEED)
modelC, resC, rawC, recC = train_model(n_unfreeze=2, loss_name="focal", epochs=6, bs=32, lr_head=1e-3, lr_vit=1e-5)
print("KET QUA (C):", {k:round(v,4) for k,v in resC.items()})
torch.save(modelC.state_dict(), "/kaggle/working/vit_finetune_focal_modelC.pt")
del modelC; torch.cuda.empty_cache()"""),

("markdown", """## So sánh & lưu kết quả — bảng tổng hợp + dự đoán từng mẫu + recall theo lớp + model (C)

Lưu đầy đủ để phân tích lỗi: bảng tổng hợp 3 cấu hình, dự đoán từng mẫu (raw
predictions) và recall theo từng nhãn cho CẢ 3 cấu hình, cùng trọng số mô hình (C)
(cấu hình fine-tune+Focal — dùng cho phân tích lỗi/so sánh sau này)."""),

("code", """import pandas as pd
res=pd.DataFrame([
 {"cấu hình":"(A) ViT-L/14 frozen + MLP (linear-probe)",**resA},
 {"cấu hình":"(B) ViT-L/14 fine-tune 2 block cuối",**resB},
 {"cấu hình":"(C) ViT-L/14 fine-tune + Focal",**resC},
]).round(4)
res.to_csv("/kaggle/working/finetune_results.csv",index=False)
print(res.to_string(index=False))

for tag, raw_df, rec_df in [("A",rawA,recA), ("B",rawB,recB), ("C",rawC,recC)]:
    raw_df.to_csv(f"/kaggle/working/finetune_{tag}_raw.csv", index=False)
    rec_df.to_csv(f"/kaggle/working/finetune_{tag}_per_class_recall.csv", index=False)

print("\\nDa luu: finetune_results.csv, finetune_{A,B,C}_raw.csv, "
      "finetune_{A,B,C}_per_class_recall.csv, vit_finetune_focal_modelC.pt")"""),

("markdown", """## Đóng gói kết quả để tải về"""),

("code", """import os, glob, zipfile
from IPython.display import FileLink
OUT="/kaggle/working/finetune_ket_qua.zip"
if os.path.exists(OUT): os.remove(OUT)
with zipfile.ZipFile(OUT,"w",zipfile.ZIP_DEFLATED) as zf:
    for f in glob.glob("/kaggle/working/finetune_*.csv") + glob.glob("/kaggle/working/finetune_*.pt") \\
            + glob.glob("/kaggle/working/vit_finetune_focal_modelC.pt"):
        zf.write(f, os.path.basename(f)); print("+", os.path.basename(f))
print(f"\\nDa tao {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
FileLink("finetune_ket_qua.zip")"""),
]
make_notebook(cells, os.path.join(HERE, "nb_finetune_kaggle.ipynb"))

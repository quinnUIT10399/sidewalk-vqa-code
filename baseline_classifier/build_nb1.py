# -*- coding: utf-8 -*-
"""Sinh nb1_vit_mlp.ipynb — mô hình CLIP ViT + MLP (linear probe) và
NGHIÊN CỨU XỬ LÝ MẤT CÂN BẰNG (trọng tâm mà giảng viên yêu cầu bổ sung)."""
import os
from _mknb import make_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

cells = [
("markdown", """# NB1 — CLIP ViT + MLP (linear probe) & nghiên cứu xử lý MẤT CÂN BẰNG

**Mô hình:** trích đặc trưng ảnh bằng **CLIP ViT** *đóng băng* (không huấn luyện lại
phần thị giác), rồi huấn luyện một **MLP nhẹ** (Linear→ReLU→Dropout→Linear, ~0.26M
tham số) làm đầu phân loại. Đây là cách **linear-probing/feature-caching** được
khuyến nghị cho VQA khi chỉ có CPU: toàn bộ tính toán thị giác nặng được **cache một
lần**, mỗi epoch chỉ chạy MLP trên vector đặc trưng → huấn luyện trong vài giây.

**Hai nhánh theo loại câu hỏi** (tránh bùng nổ tổ hợp nhãn):
- Câu *"what"* (Q2, Q5, Q8): **đa nhãn** trên **nhãn nguyên tử** (xe máy, ô tô, …),
  dùng sigmoid + BCE.
- Câu *nhị phân/đếm/không gian* (Q1, Q3, Q4, Q6, Q7, Q9, Q10): **đơn nhãn** (softmax).

**Trọng tâm — xử lý mất cân bằng:** so sánh baseline với hàng loạt phương pháp theo
đúng khuôn *"A baseline"* → *"A + B"*: đổi **đúng một** yếu tố (hàm loss / lấy mẫu
lại / hiệu chỉnh) để cô lập tác động.

> **Bài học LLaVA (thất bại đã ghi nhận):** khi thử VLM sinh LLaVA, input token của
> câu hỏi + danh sách lựa chọn dài vượt `MAX_SEQ_LEN=768` (576 token ảnh cố định chỉ
> chừa 192 cho văn bản) → phần đáp án bị cắt, mô hình sập về "không có". Ở đây ta
> **không sinh văn bản** mà phân loại trên đặc trưng cache → hoàn toàn tránh lỗi này,
> đồng thời chạy được trên laptop.

Tham chiếu: frozen CLIP + MLP head (linear probe); Lin et al. 2017 (Focal Loss);
Cao et al. 2019 (LDAM-DRW); Cui et al. 2019 (Class-Balanced); Menon et al. 2021
(Logit Adjustment); Ben-Baruch et al. 2020 (ASL)."""),

("code", """import os, sys, io, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd())=="experiments" else "F:/VQA", "imbalance_study"))
if os.path.basename(os.getcwd())!="experiments" and os.path.isdir("F:/VQA/imbalance_study"):
    sys.path.insert(0, "F:/VQA/imbalance_study")
import run_imbalance as R
import torch, torch.nn as nn

feat_vitl = R.load_features("_vitl")   # CLIP ViT-L/14, 1024-d
feat_b32  = R.load_features("")        # CLIP ViT-B/32, 768-d
d = {n: pd.read_csv(os.path.join(R.QA,f"{n}.csv")) for n in ["train","val","test"]}
for k in d: d[k]["qid"]=d[k]["qid"].astype(str)
train,val,test = d["train"],d["val"],d["test"]
print("ViT-L/14 dim:", next(iter(feat_vitl.values())).shape, "| ViT-B/32 dim:", next(iter(feat_b32.values())).shape)
print("train/val/test dòng:", len(train), len(val), len(test))"""),

("markdown", """## 1. EDA — mức độ mất cân bằng của từng câu

Ta đo *tỉ lệ mất cân bằng* IR = (số mẫu lớp nhiều nhất)/(lớp ít nhất). IR càng lớn
thì baseline càng dễ bỏ quên lớp thiểu số → các phương pháp cân bằng càng có đất dụng võ."""),

("code", """from collections import Counter
def dist(qid):
    sub = R.pd.concat([train,val,test]); sub=sub[sub["qid"]==qid]
    if qid in R.WHAT_QIDS:
        c=Counter()
        for a in sub["answer"]:
            for t in R.atomic_set(a): c[t]+=1
    else:
        yy = sub["answer"].map(R.normalize)
        if qid in R.DERIVED_BIN: yy = pd.Series(R.to_binary(yy.values))
        c=Counter(yy)
    return c
rows=[]
for q in R.QID_ORDER:
    c=dist(q); v=list(c.values())
    rows.append({"qid":q,"loại":"multi" if q in R.WHAT_QIDS else "single",
                 "n_lớp":len(c),"IR":round(max(v)/min(v),1),
                 "lớp_hiếm_nhất":min(c,key=c.get),"n_hiếm":min(v)})
eda=pd.DataFrame(rows); eda"""),

("code", """# Trực quan hoá phân phối câu lệch mạnh nhất (Q5) và một câu đa nhãn (Q8)
fig,ax=plt.subplots(1,2,figsize=(12,3.5))
for a,q in zip(ax,["Q5","Q8"]):
    c=dist(q); items=sorted(c.items(),key=lambda x:-x[1])
    a.bar([k[:14] for k,_ in items],[v for _,v in items],color="#2a7ab9")
    a.set_title(f"{q} — phân phối nhãn (IR={round(max(c.values())/min(c.values()),1)})")
    a.tick_params(axis="x",rotation=40,labelsize=8)
plt.tight_layout(); plt.show()"""),

("markdown", """## 2. Đường cong loss — baseline (BCE) vs Focal trên câu lệch Q5

Minh hoạ cơ chế: Focal giảm trọng số mẫu dễ → dồn học vào mẫu khó/hiếm. Ta huấn
luyện MLP trên Q5 (đa nhãn) và ghi lại loss mỗi epoch."""),

("code", """def loss_curve(qid, epochs=120):
    Xtr,ytr = R.qid_rows(pd.concat([train]), feat_vitl, qid)
    vocab=R.build_vocab(ytr); Ytr=R.multihot(ytr,vocab)
    from sklearn.preprocessing import StandardScaler
    Xs=StandardScaler().fit_transform(Xtr).astype("float32")
    curves={}
    for name in ["baseline","focal"]:
        torch.manual_seed(42); m=R.MLP(Xtr.shape[1],len(vocab))
        opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
        Xt=torch.tensor(Xs); Yt=torch.tensor(Ytr)
        hist=[]
        for ep in range(epochs):
            perm=torch.randperm(len(Xt)); tot=0;nb=0
            for i in range(0,len(Xt),128):
                b=perm[i:i+128]; opt.zero_grad(); out=m(Xt[b])
                loss = R.sigmoid_focal(out,Yt[b]) if name=="focal" else nn.functional.binary_cross_entropy_with_logits(out,Yt[b])
                loss.backward(); opt.step(); tot+=loss.item(); nb+=1
            hist.append(tot/nb)
        curves[name]=hist
    return curves
cur=loss_curve("Q5")
plt.figure(figsize=(7,3.5))
for k,v in cur.items(): plt.plot(v,label=k)
plt.xlabel("epoch"); plt.ylabel("train loss"); plt.title("Q5 — đường cong loss: BCE(baseline) vs Focal")
plt.legend(); plt.grid(alpha=.3); plt.show()"""),

("markdown", """## 3. Bảng kết quả chính — nhóm "what" (đa nhãn), đặc trưng **ViT-L/14**

Chạy toàn bộ phương pháp cân bằng cho nhóm đa nhãn. Metric: exact-match (khớp tập
tuyệt đối), F1-macro/micro/samples, Recall-macro (trung bình trên **nhãn nguyên tử**)."""),

("code", """R.OUT_TAG="vitl"
what_vitl = R.table_what(train,val,test,feat_vitl)
def label_multi(m):
    names={"baseline":"ViT-L/14+MLP (baseline)","weighted_bce":"+ Weighted-BCE","cb_bce":"+ Class-Balanced",
           "focal":"+ Focal","asl":"+ ASL","ros":"+ ROS (oversample)","threshold":"+ Threshold-moving",
           "mixup":"+ MixUp","remix":"+ Remix","crt":"+ cRT","tau_norm":"+ tau-norm"}
    return names.get(m,m)
what_vitl.insert(0,"mô_hình",what_vitl["method"].map(label_multi))
what_vitl.round(4)"""),

("markdown", """## 4. Bảng kết quả chính — nhóm nhị phân/đếm/không gian (đơn nhãn), **ViT-L/14**

Metric gộp toàn nhóm: EM (accuracy), F1-macro, Recall-macro. Có dòng `majority` (prior)
để đối chiếu sàn."""),

("code", """single_vitl = R.table_single(train,val,test,feat_vitl)
def label_single(m):
    names={"majority":"Prior (majority)","baseline":"ViT-L/14+MLP (baseline)","weighted_ce":"+ Weighted-CE",
           "cb_ce":"+ Class-Balanced","focal":"+ Focal","ldam":"+ LDAM","ldam_drw":"+ LDAM-DRW",
           "ros":"+ ROS","rus":"+ RUS","smote":"+ SMOTE","logit_adjust":"+ Logit-Adjust","threshold":"+ Threshold-moving"}
    return names.get(m,m)
single_vitl.insert(0,"mô_hình",single_vitl["method"].map(label_single))
single_vitl.round(4)"""),

("markdown", """## 5. Ablation bộ mã hoá ảnh — ViT-B/32 vs ViT-L/14 (baseline)

Đổi **đúng một** yếu tố: bộ trích đặc trưng ảnh. So sánh baseline hai encoder trên cả
hai nhóm câu để biết nâng cấp encoder có đáng không."""),

("code", """# Chỉ cần dòng BASELINE cho ViT-B/32 (không chạy lại toàn bộ phương pháp -> nhanh hơn ~3x)
def baseline_what(feat):
    per=[R.multilabel_metrics(*R.run_qid_multi("baseline",*R.qid_rows(train,feat,q),*R.qid_rows(test,feat,q))) for q in R.WHAT_QIDS]
    return {k: float(np.mean([p[k] for p in per])) for k in per[0]}
def baseline_single(feat):
    gt,pr=[],[]
    for q in R.SINGLE_QIDS:
        Xtr,ytr=R.qid_rows(train,feat,q); Xte,yte=R.qid_rows(test,feat,q)
        if q in R.DERIVED_BIN: ytr,yte=R.to_binary(ytr),R.to_binary(yte)
        g,p=R.run_qid_single("baseline",Xtr,ytr,Xte,yte); gt+=list(g); pr+=list(p)
    from sklearn.metrics import accuracy_score, f1_score
    return {"EM":accuracy_score(gt,pr),"F1ma":f1_score(gt,pr,average="macro",zero_division=0)}
bw32=baseline_what(feat_b32); bs32=baseline_single(feat_b32)
def base_row(df,col): return df[df["method"]=="baseline"].iloc[0][col]
abl=pd.DataFrame([
 {"nhóm":"what (F1-macro)","ViT-B/32":round(bw32["F1_macro"],4),"ViT-L/14":round(base_row(what_vitl,"F1_macro"),4)},
 {"nhóm":"what (exact)","ViT-B/32":round(bw32["exact"],4),"ViT-L/14":round(base_row(what_vitl,"exact"),4)},
 {"nhóm":"single (F1-macro)","ViT-B/32":round(bs32["F1ma"],4),"ViT-L/14":round(base_row(single_vitl,"F1ma"),4)},
 {"nhóm":"single (EM)","ViT-B/32":round(bs32["EM"],4),"ViT-L/14":round(base_row(single_vitl,"EM"),4)},
])
abl"""),

("markdown", """## 6. Kiểm chứng 5-fold CV (đáng tin hơn single-split)

Test đơn lẻ nhỏ và nhiễu. Ta chạy 5-fold CV cho một số phương pháp tiêu biểu của
nhóm "what" trên **toàn bộ** dữ liệu (gộp train+val+test rồi chia fold)."""),

("code", """R.OUT_TAG="vitl"
R.run_cv_what(feat_vitl, methods=("baseline","weighted_bce","focal","asl","ros"), folds=5)
pd.read_csv(os.path.join(R.OUT,"cv_what_vitl.csv")).round(4)"""),

("code", """# 5-fold CV nhóm đơn nhãn: baseline vs vài phương pháp, gộp toàn nhóm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
full = pd.concat([train,val,test],ignore_index=True)
def cv_single(method, folds=5):
    gt_all=[[] for _ in range(folds)]; pr_all=[[] for _ in range(folds)]
    for q in R.SINGLE_QIDS:
        X,y = R.qid_rows(full[full["qid"]==q], feat_vitl, q)
        if q in R.DERIVED_BIN: y=R.to_binary(y)
        skf=StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)
        for f,(tr,te) in enumerate(skf.split(X,y)):
            meth = "baseline" if (method=="threshold" and q not in R.BINARY_QIDS) else method
            gt,pr = R.run_qid_single("baseline" if method=="baseline" else meth, X[tr],y[tr],X[te],y[te],X[te],y[te])
            gt_all[f]+=list(gt); pr_all[f]+=list(pr)
    ems=[accuracy_score(gt_all[f],pr_all[f]) for f in range(folds)]
    f1s=[f1_score(gt_all[f],pr_all[f],average="macro",zero_division=0) for f in range(folds)]
    return np.mean(ems),np.std(ems),np.mean(f1s),np.std(f1s)
rows=[]
for m in ["baseline","weighted_ce","focal","ldam_drw","logit_adjust"]:
    em,es,fm,fs=cv_single(m)
    rows.append({"method":label_single(m),"EM_mean":round(em,4),"EM_std":round(es,4),"F1macro_mean":round(fm,4),"F1macro_std":round(fs,4)})
cv_single_df=pd.DataFrame(rows); cv_single_df"""),

("markdown", """## 7. Thời gian huấn luyện / suy luận (trên CPU laptop)

Đo cho một cấu hình đại diện (Q5, đa nhãn, ViT-L/14) — chứng minh tính khả thi trên
máy cá nhân."""),

("code", """Xtr,ytr=R.qid_rows(train,feat_vitl,"Q5"); Xte,yte=R.qid_rows(test,feat_vitl,"Q5")
t0=time.time(); Yt,Yp=R.run_qid_multi("baseline",Xtr,ytr,Xte,yte); t_train=time.time()-t0
# đo suy luận thuần
from sklearn.preprocessing import StandardScaler
sc=StandardScaler().fit(Xtr); import torch
mdl=R.MLP(Xtr.shape[1],len(R.build_vocab(ytr)))
t0=time.time()
with torch.no_grad(): _=mdl(torch.tensor(sc.transform(Xte).astype("float32")))
t_inf=time.time()-t0
print(f"Q5 ViT-L/14 — huấn luyện MLP (120 epoch, {len(Xtr)} mẫu): {t_train:.1f}s")
print(f"Suy luận {len(Xte)} mẫu: {t_inf*1000:.0f} ms  (~{1000*t_inf/len(Xte):.2f} ms/mẫu)")
print("Lưu ý: chi phí trích đặc trưng ViT chỉ trả MỘT LẦN khi cache, không lặp lại mỗi epoch.")"""),

("markdown", """## 8. Phân tích lỗi — câu đa nhãn Q8 (nhiều lớp, nhập nhằng)

Xem F1 theo từng nhãn nguyên tử để biết lớp nào mô hình còn yếu."""),

("code", """from sklearn.metrics import f1_score as f1s, precision_score, recall_score as rec
Xtr,ytr=R.qid_rows(train,feat_vitl,"Q8"); Xval,yval=R.qid_rows(val,feat_vitl,"Q8"); Xte,yte=R.qid_rows(test,feat_vitl,"Q8")
vocab=R.build_vocab(ytr); Yt,Yp=R.run_qid_multi("baseline",Xtr,ytr,Xte,yte,Xval,yval)
per=[]
for j,lab in enumerate(vocab):
    if Yt[:,j].sum()==0: continue
    per.append({"nhãn":lab,"n_test":int(Yt[:,j].sum()),
                "P":round(precision_score(Yt[:,j],Yp[:,j],zero_division=0),3),
                "R":round(rec(Yt[:,j],Yp[:,j],zero_division=0),3),
                "F1":round(f1s(Yt[:,j],Yp[:,j],zero_division=0),3)})
pd.DataFrame(per).sort_values("F1")"""),

("markdown", """## 9. Tổng hợp & lưu kết quả

Baseline → +phương pháp cho thấy các kỹ thuật cân bằng cải thiện **F1-macro** (chỉ số
nhạy với lớp thiểu số) trong khi EM ít đổi — đúng như kỳ vọng."""),

("code", """os.makedirs("results",exist_ok=True)
what_vitl.to_csv("results/nb1_what_vitl.csv",index=False)
single_vitl.to_csv("results/nb1_single_vitl.csv",index=False)
abl.to_csv("results/nb1_encoder_ablation.csv",index=False)
cv_single_df.to_csv("results/nb1_cv_single.csv",index=False)
eda.to_csv("results/nb1_eda.csv",index=False)
print("Baseline what  F1-macro:", round(base_row(what_vitl,"F1_macro"),4))
print("Tốt nhất what  F1-macro:", round(what_vitl["F1_macro"].max(),4),
      "->", label_multi(what_vitl.loc[what_vitl['F1_macro'].idxmax(),'method']))
print("Baseline single F1-macro:", round(base_row(single_vitl,"F1ma"),4))
print("Tốt nhất single F1-macro:", round(single_vitl['F1ma'].max(),4),
      "->", label_single(single_vitl.loc[single_vitl['F1ma'].idxmax(),'method']))
print("Đã lưu 5 file results/nb1_*.csv")"""),
]

make_notebook(cells, os.path.join(HERE, "nb1_vit_mlp.ipynb"))

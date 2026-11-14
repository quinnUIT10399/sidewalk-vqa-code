# -*- coding: utf-8 -*-
"""Sinh nb2_phobert_vit_shared.ipynb — MO HINH DUNG-CHUNG PhoBERT+ViT,
cham diem theo DUNG chuan per-question atomic (giong nb1)."""
import os
from _mknb import make_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

cells = [
("markdown", """# NB2 — VQA dùng-chung: **PhoBERT (câu hỏi) + ViT (ảnh)**, một model cho cả 10 câu

**Mô hình:** một model **duy nhất** cho toàn bộ 10 câu (multi-task). Đặc trưng hợp
nhất `x = [PhoBERT_meanpool(câu hỏi) ‖ ViT-L/14(ảnh)]` (768+1024 = 1792-d) → MLP →
**sigmoid trên GLOBAL ATOM VOCAB** (hợp nhất mọi nhãn nguyên tử của cả 10 câu).
Đây là baseline VQA "thật" theo công thức OpenViVQA/ViVQA (PhoBERT + ViT + hợp nhất).

**Điểm mấu chốt — chấm điểm theo ĐÚNG chuẩn per-question atomic (giống hệt nb1):**
- Câu *what* (Q2/Q5/Q8): decode = **ngưỡng** trên tập atom *ứng viên của câu đó* →
  exact-match, F1-macro/micro/samples, Recall-macro (đa nhãn nguyên tử).
- Câu nhị phân/đếm/không gian: decode = **argmax** trên atom ứng viên của câu → EM, F1-macro.

→ Không còn khung "combo-class toàn cục" cũ (đã bỏ). Mọi model (Prior, ViT+MLP, model
này) giờ **cùng một thước đo** nên xếp chung một bảng so sánh được.

> **Bài học LLaVA:** PhoBERT ở đây là **bộ mã hoá** (không sinh), câu hỏi ngắn cắt
> `max_length=64`, và chỉ có 10 câu hỏi cố định nên cache một lần → tuyệt đối không
> gặp lỗi tràn token như LLaVA (câu hỏi + lựa chọn dài vượt giới hạn, đáp án bị cắt).

Tham chiếu: Nguyen et al. 2023 (OpenViVQA); Tran et al. 2021 (ViVQA); PhoBERT
(Nguyen & Nguyen 2020); Lin et al. 2017 (Focal); Ben-Baruch et al. 2020 (ASL)."""),

("code", """import os, sys, io, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
if os.path.basename(os.getcwd())!="experiments" and os.path.isdir("F:/VQA/experiments"):
    os.chdir("F:/VQA/experiments")
import unified as U
vit=U.load_vit("_vitl"); pho=U.load_phobert()
tr=pd.read_csv(os.path.join(U.QA,"train.csv")); tr["qid"]=tr["qid"].astype(str)
va=pd.read_csv(os.path.join(U.QA,"val.csv"));   va["qid"]=va["qid"].astype(str)
te=pd.read_csv(os.path.join(U.QA,"test.csv"));  te["qid"]=te["qid"].astype(str)
full=pd.concat([tr,va,te],ignore_index=True)
vocab,idx,cand=U.build_global_vocab(full)
Xtr,Ytr,qtr,rtr=U.build_rows(tr,vit,pho,idx,"full")
Xte,Yte,qte,rte=U.build_rows(te,vit,pho,idx,"full")
Xva,Yva,qva,rva=U.build_rows(va,vit,pho,idx,"full")
print("Đặc trưng hợp nhất:", Xtr.shape, "| GLOBAL atom vocab:", len(vocab))"""),

("markdown", """## 1. EDA — GLOBAL atom vocab & tập atom ứng viên mỗi câu (cơ chế định tuyến)

Model học chung một không gian nhãn nguyên tử; PhoBERT giúp phân biệt câu nào dùng
tập atom nào. Bảng dưới cho thấy "đường ray" quyết định của từng câu."""),

("code", """print("GLOBAL atom vocab (%d):"%len(vocab)); print(vocab)
rows=[]
for q in U.QID_ORDER:
    rows.append({"qid":q,"loại":"multi" if q in U.WHAT_QIDS else "single",
                 "atom_ứng_viên":", ".join(vocab[i] for i in cand[q])})
pd.DataFrame(rows)"""),

("markdown", """## 2. Baseline — model đầy đủ (Q+I), chi tiết theo từng câu"""),

("code", """t0=time.time()
model,sc,pos=U.train_shared(Xtr,Ytr,"baseline",epochs=80)
pred=U.predict_per_question(model,sc,Xte,qte,cand,"baseline")
t_base=time.time()-t0
w,s,_=U.score_tables(pred,Yte,qte,cand,vocab)
print("Baseline — what F1-macro=%.3f exact=%.3f | single F1-macro=%.3f EM=%.3f  (train+infer %.0fs)"%(
    w["F1_macro"],w["exact"],s["F1ma"],s["EM"],t_base))
U.per_question_breakdown(pred,Yte,qte,cand,vocab)"""),

("markdown", """## 3. Thực nghiệm nhiều phương pháp cân bằng ("A baseline" → "A + B")

Giữ nguyên kiến trúc PhoBERT+ViT dùng-chung, chỉ đổi **đúng một** yếu tố (hàm loss/
hiệu chỉnh/ngưỡng). Báo cáo cả hai nhóm câu theo chuẩn per-question atomic."""),

("code", """def label(m): return {"baseline":"PhoBERT+ViT-L (baseline)","weighted_bce":"+ Weighted-BCE",
        "focal":"+ Focal","asl":"+ ASL","logit_adjust":"+ Logit-Adjust","threshold":"+ Threshold-moving"}[m]
rows=[]; preds={}
for m in ["baseline","weighted_bce","focal","asl","logit_adjust","threshold"]:
    model,sc,pos=U.train_shared(Xtr,Ytr,m,epochs=80)
    p=U.predict_per_question(model,sc,Xte,qte,cand,m,pos_train=pos,Xval=Xva,Yval=Yva,qsval=qva)
    preds[m]=p; w,s,_=U.score_tables(p,Yte,qte,cand,vocab)
    rows.append({"mô_hình":label(m),
                 "what_exact":round(w["exact"],4),"what_F1macro":round(w["F1_macro"],4),
                 "what_F1micro":round(w["F1_micro"],4),"what_Rmacro":round(w["R_macro"],4),
                 "single_EM":round(s["EM"],4),"single_F1macro":round(s["F1ma"],4),"single_Rmacro":round(s["Rma"],4)})
shared_tbl=pd.DataFrame(rows); shared_tbl"""),

("markdown", """## 4. Ablation đầu vào — đóng góp của PhoBERT

Đổi đúng một yếu tố: có/không nhánh câu hỏi PhoBERT trong đầu vào (giữ nguyên cách
decode theo atom ứng viên của câu). So sánh cho biết đặc trưng câu hỏi PhoBERT đóng
góp thêm bao nhiêu ngoài việc đã biết tập atom ứng viên."""),

("code", """rows=[]
for mode in ["full","i_only"]:
    Xa,Ya,qa,_=U.build_rows(tr,vit,pho,idx,mode)
    Xb,Yb,qb,_=U.build_rows(te,vit,pho,idx,mode)
    model,sc,pos=U.train_shared(Xa,Ya,"baseline",epochs=80)
    p=U.predict_per_question(model,sc,Xb,qb,cand,"baseline")
    w,s,_=U.score_tables(p,Yb,qb,cand,vocab)
    rows.append({"đầu_vào":{"full":"Q+I (PhoBERT+ViT)","i_only":"I-only (bỏ PhoBERT)"}[mode],
                 "what_F1macro":round(w["F1_macro"],4),"single_F1macro":round(s["F1ma"],4),
                 "single_EM":round(s["EM"],4)})
abl=pd.DataFrame(rows); abl"""),

("markdown", """## 5. Kiểm chứng 5-fold CV — baseline vs phương pháp tốt nhất

Gộp toàn bộ dữ liệu, chia 5-fold theo dòng. Báo cáo trung bình ± lệch (what F1-macro,
single F1-macro)."""),

("code", """from sklearn.model_selection import KFold
Xall,Yall,qall,_=U.build_rows(full,vit,pho,idx,"full")
best_key = {v:k for k,v in {"baseline":"PhoBERT+ViT-L (baseline)","weighted_bce":"+ Weighted-BCE","focal":"+ Focal","asl":"+ ASL","logit_adjust":"+ Logit-Adjust","threshold":"+ Threshold-moving"}.items()}[
    shared_tbl.loc[shared_tbl["what_F1macro"].idxmax(),"mô_hình"]]
def cv_shared(method,folds=5):
    kf=KFold(n_splits=folds,shuffle=True,random_state=42); wf=[];sf=[]
    for trn,tst in kf.split(Xall):
        model,sc,pos=U.train_shared(Xall[trn],Yall[trn],method,epochs=80)
        p=U.predict_per_question(model,sc,Xall[tst],qall[tst],cand,method,pos_train=pos)
        w,s,_=U.score_tables(p,Yall[tst],qall[tst],cand,vocab); wf.append(w["F1_macro"]); sf.append(s["F1ma"])
    return np.mean(wf),np.std(wf),np.mean(sf),np.std(sf)
rows=[]
for m in dict.fromkeys(["baseline",best_key]):
    wm,ws,sm,ss=cv_shared(m)
    rows.append({"mô_hình":label(m),"what_F1macro_mean":round(wm,4),"what_F1macro_std":round(ws,4),
                 "single_F1macro_mean":round(sm,4),"single_F1macro_std":round(ss,4)})
cv_shared_df=pd.DataFrame(rows); cv_shared_df"""),

("markdown", """## 6. So sánh với NB1 (đầu riêng cho từng câu) — cùng một thước đo

Vì giờ **cùng chuẩn per-question atomic**, ta đặt thẳng model dùng-chung cạnh model
ViT+MLP có đầu riêng cho từng câu (NB1). Kỳ vọng: đầu riêng nhỉnh hơn một chút, nhưng
model dùng-chung là **một mô hình VQA multimodal thật** (có PhoBERT) — đánh đổi giữa
tính gọn (1 model) và điểm số."""),

("code", """try:
    nb1w=pd.read_csv("results/nb1_what_vitl.csv"); nb1s=pd.read_csv("results/nb1_single_vitl.csv")
    base_nb1_what=float(nb1w[nb1w["method"]=="baseline"]["F1_macro"])
    base_nb1_single=float(nb1s[nb1s["method"]=="baseline"]["F1ma"])
    cmp=pd.DataFrame([
      {"model":"ViT-L/14 + MLP (đầu riêng từng câu, NB1)","what_F1macro":round(base_nb1_what,4),"single_F1macro":round(base_nb1_single,4)},
      {"model":"PhoBERT + ViT-L dùng-chung (NB2)","what_F1macro":round(w["F1_macro"],4),"single_F1macro":round(s["F1ma"],4)},
    ])
    display(cmp)
except Exception as e:
    print("(chưa có kết quả NB1 để so sánh:",e,")")"""),

("markdown", """## 7. Tổng hợp & lưu kết quả"""),

("code", """os.makedirs("results",exist_ok=True)
shared_tbl.to_csv("results/nb2_shared_methods.csv",index=False)
abl.to_csv("results/nb2_input_ablation.csv",index=False)
cv_shared_df.to_csv("results/nb2_cv.csv",index=False)
U.per_question_breakdown(preds["baseline"],Yte,qte,cand,vocab).to_csv("results/nb2_per_question.csv",index=False)
print("Baseline what F1-macro   =", shared_tbl.iloc[0]["what_F1macro"])
print("Tốt nhất what F1-macro   =", shared_tbl["what_F1macro"].max(),
      "->", shared_tbl.loc[shared_tbl['what_F1macro'].idxmax(),'mô_hình'])
print("Baseline single F1-macro =", shared_tbl.iloc[0]["single_F1macro"])
print("Đã lưu 4 file results/nb2_*.csv")"""),
]

make_notebook(cells, os.path.join(HERE, "nb2_phobert_vit_shared.ipynb"))

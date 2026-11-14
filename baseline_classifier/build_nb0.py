# -*- coding: utf-8 -*-
"""Sinh nb0_prior.ipynb — baseline THIEN VI DA SO (majority-class prior)."""
import os
from _mknb import make_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

cells = [
("markdown", """# NB0 — Baseline "Prior" (thiên vị đa số / majority-class)

**Vai trò:** đây là *sàn tham chiếu* (floor) cho toàn bộ thực nghiệm. Mô hình không
học gì từ ảnh: với mỗi câu hỏi nó luôn đoán **đáp án phổ biến nhất trong tập train**.

Trong tài liệu VQA, baseline này còn gọi là **"language prior"** — cho thấy mức
điểm mà một mô hình có thể đạt được *chỉ nhờ đoán mù theo phân phối đáp án*, tức là
mức "thiên vị ngôn ngữ" của bộ dữ liệu. Mọi mô hình học thật (NB1, NB2) phải vượt
qua mốc này thì mới chứng minh được là ảnh/đặc trưng thực sự có ích.

Tham chiếu: Goyal et al. 2017 (VQA v2, khử language-prior); "Loss re-scaling VQA"
(arXiv 2010.16010) diễn giải prior như bài toán mất cân bằng lớp."""),

("code", """import os, re, sys, io
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score

QA = os.path.join(os.path.dirname(os.getcwd()), "qa") if os.path.basename(os.getcwd())=="experiments" else "F:/VQA/qa"
QID_ORDER = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10"]
WHAT = {"Q2","Q5","Q8"}   # câu đa nhãn (multi-label)
SEED = 42

def norm(s): return re.sub(r"\\s+"," ",str(s).strip().lower())

full = pd.concat([pd.read_csv(os.path.join(QA,f"{s}.csv")) for s in ["train","val","test"]], ignore_index=True)
full["qid"] = full["qid"].astype(str)
train = pd.read_csv(os.path.join(QA,"train.csv")); train["qid"]=train["qid"].astype(str)
test  = pd.read_csv(os.path.join(QA,"test.csv"));  test["qid"]=test["qid"].astype(str)
print("Tổng:", len(full), "dòng /", full["filename"].nunique(), "ảnh")"""),

("markdown", """## 1. EDA — đáp án đa số của mỗi câu

Prior chỉ cần biết đáp án nào chiếm đa số trong train. Bảng dưới cho thấy đáp án đó
và tỉ lệ của nó — cũng chính là *accuracy mà prior sẽ đạt* nếu test có cùng phân phối."""),

("code", """rows=[]
for q in QID_ORDER:
    sub = train[train["qid"]==q]
    vc = sub["answer"].map(norm).value_counts()
    rows.append({"qid":q, "n_train":len(sub), "đáp_án_đa_số":vc.index[0],
                 "tỉ_lệ":round(vc.iloc[0]/len(sub),3)})
pd.DataFrame(rows)"""),

("markdown", """## 2. Đánh giá prior trên tập test (single-split)

Với mỗi câu, đoán đáp án đa số của train cho *mọi* ảnh test rồi đo EM (accuracy) và
F1-macro. Câu đa nhãn (Q2/Q5/Q8): prior = tập nhãn đa số (thường là "không có")."""),

("code", """def prior_eval(qids, df_tr, df_te):
    out=[]
    for q in qids:
        maj = df_tr[df_tr["qid"]==q]["answer"].map(norm).value_counts().index[0]
        yte = df_te[df_te["qid"]==q]["answer"].map(norm).values
        yp  = np.array([maj]*len(yte), object)
        out.append({"qid":q,"n_test":len(yte),
                    "EM":round(accuracy_score(yte,yp),4),
                    "F1_macro":round(f1_score(yte,yp,average="macro",zero_division=0),4),
                    "R_macro":round(recall_score(yte,yp,average="macro",zero_division=0),4)})
    return pd.DataFrame(out)

tbl = prior_eval(QID_ORDER, train, test)
tbl.loc["TB"] = ["—","—",round(tbl["EM"].mean(),4),round(tbl["F1_macro"].mean(),4),round(tbl["R_macro"].mean(),4)]
tbl"""),

("markdown", """## 3. Kiểm chứng bằng 5-fold CV (đáng tin hơn single-split)

Test nhỏ nên single-split nhiễu. Ta lặp prior trên 5-fold: mỗi fold lấy đáp án đa số
từ phần train của fold rồi đo trên phần test của fold. Báo cáo trung bình ± độ lệch."""),

("code", """def prior_cv(qids, folds=5):
    out=[]
    for q in qids:
        sub = full[full["qid"]==q].reset_index(drop=True)
        y = sub["answer"].map(norm).values
        if len(set(y))<2:
            out.append({"qid":q,"EM_mean":round((y==pd.Series(y).mode()[0]).mean(),4),"EM_std":0.0,
                        "F1macro_mean":0.0,"F1macro_std":0.0}); continue
        skf=StratifiedKFold(n_splits=folds,shuffle=True,random_state=SEED)
        ems,f1s=[],[]
        for tr,te in skf.split(np.zeros(len(y)),y):
            maj=pd.Series(y[tr]).value_counts().index[0]
            yp=np.array([maj]*len(te),object)
            ems.append(accuracy_score(y[te],yp)); f1s.append(f1_score(y[te],yp,average="macro",zero_division=0))
        out.append({"qid":q,"EM_mean":round(np.mean(ems),4),"EM_std":round(np.std(ems),4),
                    "F1macro_mean":round(np.mean(f1s),4),"F1macro_std":round(np.std(f1s),4)})
    return pd.DataFrame(out)

cv = prior_cv(QID_ORDER)
print("Prior 5-fold CV — trung bình EM   =", round(cv["EM_mean"].mean(),4))
print("Prior 5-fold CV — trung bình F1ma =", round(cv["F1macro_mean"].mean(),4))
cv"""),

("markdown", """## 4. Kết luận NB0

- Prior đạt **EM cao** ở các câu lệch mạnh (Q1, Q7...) chỉ nhờ đoán đáp án đa số,
  nhưng **F1-macro rất thấp** vì hoàn toàn bỏ lớp thiểu số → đúng bản chất bài toán
  mất cân bằng.
- Đây là mốc sàn: NB1 (ViT+MLP) và NB2 (PhoBERT+ViT) phải vượt **F1-macro** này để
  chứng minh giá trị của đặc trưng ảnh và của các phương pháp cân bằng.

Lưu bảng kết quả để dùng cho báo cáo Chương 5."""),

("code", """os.makedirs("results", exist_ok=True)
tbl.to_csv("results/nb0_prior_split.csv", index=False)
cv.to_csv("results/nb0_prior_cv.csv", index=False)
print("Đã lưu results/nb0_prior_split.csv và results/nb0_prior_cv.csv")"""),
]

make_notebook(cells, os.path.join(HERE, "nb0_prior.ipynb"))

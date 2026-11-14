# -*- coding: utf-8 -*-
"""
MO HINH VQA DUNG-CHUNG (shared multi-task) — PhoBERT(cau hoi) + ViT(anh).

MOT model duy nhat cho CA 10 CAU. Embedding cau hoi PhoBERT dinh tuyen "dang hoi
cau nao"; anh ViT cung cap noi dung. Dau ra: sigmoid tren GLOBAL ATOM VOCAB (hop
nhat moi nhan nguyen tu cua toan bo 10 cau).

CHAM DIEM theo DUNG CHUAN per-question atomic (giong nb1), KHONG dung combo cu:
  * Cau "what" (Q2,Q5,Q8): da nhan -> nguong trong tap atom UNG VIEN cua cau do.
  * Cau don nhan (nhi phan/dem/khong gian): argmax trong tap atom ung vien cua cau.
Metric tai dung run_imbalance.multilabel_metrics / pooled_single -> so sanh 1-1 voi nb1.

Bien the dau vao (ablation): full (Q+I) | i_only (bo PhoBERT) | q_only (bo anh).
Phuong phap mat can bang: baseline(BCE) | weighted_bce | focal | asl | logit_adjust | threshold.
"""
import os, re, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(HERE)
QA    = os.path.join(ROOT, "qa")
CACHE = os.path.join(HERE, "cache")
VIT_CACHE = os.path.join(ROOT, "imbalance_study", "cache")
sys.path.insert(0, os.path.join(ROOT, "imbalance_study"))
import run_imbalance as R   # tai dung: normalize, atomic_set, to_binary, metrics

SEED = 42
DEVICE = "cpu"
QID_ORDER   = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10"]
WHAT_QIDS   = ["Q2","Q5","Q8"]
SINGLE_QIDS = ["Q1","Q3","Q4","Q6","Q7","Q9","Q10"]
DERIVED_BIN = {"Q4"}
np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(os.cpu_count() or 4)


# ───────── du lieu / dac trung ─────────
def load_vit(tag="_vitl"):
    X = np.load(os.path.join(VIT_CACHE, f"features{tag}.npy"))
    names = open(os.path.join(VIT_CACHE, f"filenames{tag}.txt"), encoding="utf-8").read().splitlines()
    return {n: X[i] for i, n in enumerate(names)}

def load_phobert():
    X = np.load(os.path.join(CACHE, "phobert_q.npy"))
    qids = open(os.path.join(CACHE, "phobert_qids.txt"), encoding="utf-8").read().splitlines()
    return {q: X[i] for i, q in enumerate(qids)}

def answer_atoms(qid, ans):
    """Tap atom cua 1 dap an, ap dung quy tac Q4 (dan xuat nhi phan)."""
    if qid in DERIVED_BIN:
        return {R.to_binary([ans])[0]}
    return R.atomic_set(ans)


def build_global_vocab(df):
    """GLOBAL atom vocab + tap atom ung vien cho tung cau."""
    vocab = set(); cand = {q: set() for q in QID_ORDER}
    for _, r in df.iterrows():
        q = str(r["qid"]); ats = answer_atoms(q, r["answer"])
        vocab |= ats; cand[q] |= ats
    vocab = sorted(vocab)
    idx = {a: i for i, a in enumerate(vocab)}
    cand_idx = {q: sorted(idx[a] for a in cand[q]) for q in QID_ORDER if cand[q]}
    return vocab, idx, cand_idx


def build_rows(df, vit, pho, idx, mode="full"):
    """Tra ve X (dac trung hop nhat), Y (multi-hot global), qs (qid), raw (dap an chuan hoa)."""
    di = next(iter(vit.values())).shape[0]; dq = next(iter(pho.values())).shape[0]
    A = len(idx); Xr, Ys, qs, raw = [], [], [], []
    for _, r in df.iterrows():
        q = str(r["qid"]); img = vit.get(r["filename"]); qv = pho.get(q)
        if img is None or qv is None:
            continue
        if mode == "full":     v = np.concatenate([qv, img])
        elif mode == "i_only": v = np.concatenate([np.zeros(dq, np.float32), img])
        elif mode == "q_only": v = np.concatenate([qv, np.zeros(di, np.float32)])
        y = np.zeros(A, np.float32)
        for a in answer_atoms(q, r["answer"]):
            if a in idx: y[idx[a]] = 1.0
        Xr.append(v); Ys.append(y); qs.append(q); raw.append(R.normalize(r["answer"]))
    return (np.asarray(Xr, np.float32), np.asarray(Ys, np.float32),
            np.asarray(qs, object), np.asarray(raw, object))


class MLP(nn.Module):
    def __init__(self, d_in, n_out, hidden=512, p=0.3, n_layers=2):
        super().__init__()
        if n_layers == 1:
            self.net = nn.Linear(d_in, n_out)
        else:
            layers = [nn.Linear(d_in, hidden), nn.ReLU()]
            if p > 0: layers.append(nn.Dropout(p))
            layers.append(nn.Linear(hidden, n_out))
            self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def train_shared(Xtr, Ytr, method="baseline", epochs=80, bs=256, lr=1e-3,
                 use_scaler=True, dropout=0.3, n_layers=2, hidden=512):
    """use_scaler/dropout/n_layers/hidden: doi kien truc MANG (ablation), giu
    nguyen phuong phap mat can bang (method) khi can tach bach 2 truc thuc nghiem."""
    if use_scaler:
        sc = StandardScaler().fit(Xtr); Xs = sc.transform(Xtr).astype(np.float32)
    else:
        sc = None; Xs = Xtr.astype(np.float32)
    A = Ytr.shape[1]
    pos = Ytr.sum(0); neg = len(Ytr) - pos
    pos_weight = None
    if method == "weighted_bce":
        pos_weight = torch.tensor(np.clip(neg / np.clip(pos, 1, None), 0.1, 20), dtype=torch.float32)
    model = MLP(Xtr.shape[1], A, hidden=hidden, p=dropout, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.tensor(Xs); Yt = torch.tensor(Ytr); n = len(Xt)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i+bs]; xb, yb = Xt[b], Yt[b]
            opt.zero_grad(); out = model(xb)
            if method == "focal":  loss = R.sigmoid_focal(out, yb)
            elif method == "asl":  loss = R.asymmetric_loss(out, yb)
            else: loss = nn.functional.binary_cross_entropy_with_logits(
                        out, yb, pos_weight=pos_weight.to(DEVICE) if pos_weight is not None else None)
            loss.backward(); opt.step()
    return model, sc, pos


def _apply_scaler(sc, X):
    return sc.transform(X).astype(np.float32) if sc is not None else X.astype(np.float32)


def predict_per_question(model, sc, Xte, qs, cand_idx, method="baseline", pos_train=None,
                         Xval=None, Yval=None, qsval=None, return_proba=False):
    """Decode theo tung cau: single=argmax trong atom ung vien; what=nguong.
    return_proba=True: tra ve them P (N,A) xac suat sigmoid (dung cho AUC-ROC/PR-AUC)."""
    model.eval()
    with torch.no_grad():
        logit = model(torch.tensor(_apply_scaler(sc, Xte))).numpy()
    if method == "logit_adjust" and pos_train is not None:
        prior = pos_train / max(pos_train.sum(), 1)
        logit = logit - np.log(prior + 1e-12)[None, :]
    P = 1.0 / (1.0 + np.exp(-logit))
    A = P.shape[1]
    # nguong moi atom cho nhom what (tinh tren val neu co)
    thr = np.full(A, 0.5)
    if method == "threshold" and Xval is not None:
        with torch.no_grad():
            Pv = torch.sigmoid(model(torch.tensor(_apply_scaler(sc, Xval)))).numpy()
        from sklearn.metrics import f1_score
        for j in range(A):
            if Yval[:, j].sum() > 0:
                thr[j] = max(np.linspace(0.1, 0.9, 33),
                             key=lambda t: f1_score(Yval[:, j], (Pv[:, j] >= t).astype(int), zero_division=0))
    preds = np.zeros((len(Xte), A), dtype=int)
    for i in range(len(Xte)):
        q = qs[i]; cand = cand_idx.get(q, list(range(A)))
        scores = P[i, cand]
        if q in WHAT_QIDS:
            take = [c for c, s in zip(cand, scores) if s >= thr[c]]
            if not take: take = [cand[int(np.argmax(scores))]]
            preds[i, take] = 1
        else:
            preds[i, cand[int(np.argmax(scores))]] = 1
    if return_proba:
        return preds, P
    return preds


# ───────── cham diem theo chuan nb1 ─────────
def score_tables(preds, Ytrue, qs, cand_idx, vocab):
    """Tra ve (what_metrics_avg, single_pooled) — cung metric voi run_imbalance."""
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    inv = {i: a for i, a in enumerate(vocab)}
    # WHAT: multilabel_metrics tren subspace atom ung vien cua tung cau
    what_per = []
    for q in WHAT_QIDS:
        m = qs == q
        cand = cand_idx[q]
        Yt = Ytrue[m][:, cand]; Yp = preds[m][:, cand]
        what_per.append(R.multilabel_metrics(Yt.astype(int), Yp.astype(int)))
    what_avg = {k: float(np.mean([p[k] for p in what_per])) for k in what_per[0]}
    # SINGLE: tinh metric TUNG CAU (moi cau trong khong gian nhan RIENG) roi TRUNG
    # BINH — KHONG gop chung khong gian nhan (tranh tron nhan dem Q6 voi yes/no,
    # khien 1 cau chi phoi macro-F1). Day la cach chuan cho da-nhiem VQA.
    single_per = []
    for q in SINGLE_QIDS:
        m = qs == q; cand = cand_idx[q]
        if m.sum() == 0:
            continue
        classes = [inv[c] for c in cand]
        gt = [classes[int(np.argmax(t))] for t in Ytrue[m][:, cand]]
        pr = [classes[int(np.argmax(p))] for p in preds[m][:, cand]]
        single_per.append({
            "EM": accuracy_score(gt, pr),
            "F1ma": f1_score(gt, pr, average="macro", labels=classes, zero_division=0),
            "Rma": recall_score(gt, pr, average="macro", labels=classes, zero_division=0)})
    single = {k: float(np.mean([p[k] for p in single_per])) for k in single_per[0]}
    return what_avg, single, what_per


def per_question_breakdown(preds, Ytrue, qs, cand_idx, vocab):
    from sklearn.metrics import accuracy_score, f1_score
    inv = {i: a for i, a in enumerate(vocab)}; rows = []
    for q in QID_ORDER:
        m = qs == q
        if m.sum() == 0: continue
        cand = cand_idx[q]
        if q in WHAT_QIDS:
            mm = R.multilabel_metrics(Ytrue[m][:, cand].astype(int), preds[m][:, cand].astype(int))
            rows.append({"qid": q, "n": int(m.sum()), "loại": "multi",
                         "EM/exact": round(mm["exact"], 4), "F1_macro": round(mm["F1_macro"], 4),
                         "R_macro": round(mm["R_macro"], 4)})
        else:
            gt = [inv[cand[int(np.argmax(t))]] for t in Ytrue[m][:, cand]]
            pr = [inv[cand[int(np.argmax(p))]] for p in preds[m][:, cand]]
            rows.append({"qid": q, "n": int(m.sum()), "loại": "single",
                         "EM/exact": round(accuracy_score(gt, pr), 4),
                         "F1_macro": round(f1_score(gt, pr, average="macro", zero_division=0), 4),
                         "R_macro": round(f1_score(gt, pr, average="macro", zero_division=0), 4)})
    return pd.DataFrame(rows)

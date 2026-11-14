# -*- coding: utf-8 -*-
"""
Cache dac trung PhoBERT cho CAU HOI.

Vi bo du lieu chi co 10 cau hoi CO DINH (moi QID dung dung 1 chuoi cau hoi), ta
CHI can encode 10 chuoi mot lan roi tra cuu theo qid -> RAT nhanh tren CPU.

Xuat ra experiments/cache/phobert_q.npy (10 x 768) + phobert_qids.txt.

Bai hoc LLaVA: mo hinh sinh (generative VLM) that bai vi input token cua cau hoi +
danh sach lua chon dai vuot gioi han (MAX_SEQ_LEN=768, 576 token anh co dinh ->
chi con 192 cho van ban). O day ta dung PhoBERT lam ENCODER (khong sinh), cau hoi
ngan, va cat/pad theo max_length an toan -> khong con van de tran token.
"""
import os, sys, io
import numpy as np
import pandas as pd
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(HERE)
QA    = os.path.join(ROOT, "qa")
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)

MODEL = "vinai/phobert-base"          # 768-d, encoder tieng Viet chuan cho VQA tieng Viet
MAX_LEN = 64                          # cau hoi ngan; cat an toan (bai hoc tran token LLaVA)
QID_ORDER = ["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10"]


def main():
    from transformers import AutoTokenizer, AutoModel

    # lay 1 chuoi cau hoi dai dien cho moi QID tu dataset.csv
    df = pd.read_csv(os.path.join(QA, "dataset.csv")); df["qid"] = df["qid"].astype(str)
    questions = {}
    for q in QID_ORDER:
        sub = df[df["qid"] == q]
        if len(sub):
            questions[q] = sub["question"].mode().iloc[0]
    qids = [q for q in QID_ORDER if q in questions]
    texts = [questions[q] for q in qids]

    print(f"Encode {len(qids)} cau hoi bang {MODEL} (max_len={MAX_LEN})...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    torch.set_num_threads(os.cpu_count() or 4)

    feats = []
    with torch.no_grad():
        for q, t in zip(qids, texts):
            enc = tok(t, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding="max_length")
            out = model(**enc)
            # mean-pooling qua cac token thuc (theo attention_mask)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            vec = (summed / counts).squeeze(0).numpy().astype(np.float32)
            feats.append(vec)
            print(f"  {q:4} dim={vec.shape[0]}  '{t[:60]}...'")

    X = np.stack(feats)
    np.save(os.path.join(CACHE, "phobert_q.npy"), X)
    with open(os.path.join(CACHE, "phobert_qids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(qids))
    print(f"\nDa luu phobert_q.npy shape={X.shape} + phobert_qids.txt")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_iaa.py — Tinh IAA giua 2 annotator tren bo anh GIAO NHAU
Vietnamese Sidewalk VQA Dataset

So sanh 2 file vqa_<A>.csv va vqa_<B>.csv (do vqa_annotate.py sinh ra).
IAA tach 2 tang:
  Tang 2 (chinh) — DONG THUAN DAP AN: cung (anh, qid) -> cung dap an?
      * yes_no / choice  -> Cohen's kappa + % exact match
      * violations (G2)  -> per-nhan kappa macro + Jaccard
      * count / text     -> % exact match (sau normalize)
  Tang 1 — DONG THUAN CHON CAU: voi moi qid tuy chon, 2 nguoi co cung chon
           hoi cau do tren tung anh khong? -> Cohen's kappa nhi phan.

Dung:
  python compute_iaa.py --a dung --b ban
  python compute_iaa.py --file-a F:\\VQA\\qa\\vqa_dung.csv --file-b F:\\VQA\\qa\\vqa_ban.csv
"""

from __future__ import annotations

import csv
import argparse
import unicodedata
from pathlib import Path
from collections import defaultdict

QA_DIR = Path(r"F:\VQA\qa")

CORE_QIDS = {"G1", "G3"}   # luon hoi -> khong tinh tang-1 selection (G2 da bo)
STD_LABELS = [
    ("CAT-1", "kinh doanh / biển hiệu trên vỉa hè"),
    ("CAT-2", "vật liệu xây dựng / máy móc trên vỉa hè"),
    ("CAT-3", "để xe trái phép"),
    ("CAT-4", "xe đi trên vỉa hè"),
    ("CAT-5", "vật cản / cấu trúc chiếm dụng vỉa hè"),
    ("CAT-6", "hư hỏng hạ tầng"),
]
LABEL2CODE = {lab: code for code, lab in STD_LABELS}


def norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.rstrip(".")
    s = " ".join(s.split())
    return unicodedata.normalize("NFC", s)


def cohen_kappa(pairs) -> float | None:
    """pairs: list[(a,b)] nhan rời rạc. Tra ve kappa, hoac None neu khong tinh duoc."""
    n = len(pairs)
    if n == 0:
        return None
    labels = set()
    for a, b in pairs:
        labels.add(a); labels.add(b)
    po = sum(1 for a, b in pairs if a == b) / n
    # xac suat ky vong theo phan phoi bien
    ca = defaultdict(int); cb = defaultdict(int)
    for a, b in pairs:
        ca[a] += 1; cb[b] += 1
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else None   # khong co phuong sai -> kappa vo nghia
    return (po - pe) / (1 - pe)


def kappa_str(k) -> str:
    if k is None:
        return "n/a "
    tag = "poor"
    if k >= 0.8: tag = "almost perfect"
    elif k >= 0.6: tag = "substantial"
    elif k >= 0.4: tag = "moderate"
    elif k >= 0.2: tag = "fair"
    return f"{k:+.3f} ({tag})"


def load(path: Path) -> dict:
    """-> {(filename, qid): row}  va  {filename: set(qid)}"""
    rows = {}
    sel  = defaultdict(set)
    files = set()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fn = r.get("filename"); qid = r.get("qid")
            if not fn or not qid:
                continue
            rows[(fn, qid)] = r
            sel[fn].add(qid)
            files.add(fn)
    return {"rows": rows, "sel": sel, "files": files}


def main():
    ap = argparse.ArgumentParser(description="Tinh IAA giua 2 annotator")
    ap.add_argument("--a", help="Ten annotator A (suy ra vqa_<A>.csv)")
    ap.add_argument("--b", help="Ten annotator B (suy ra vqa_<B>.csv)")
    ap.add_argument("--file-a", help="Duong dan file A (ghi de --a)")
    ap.add_argument("--file-b", help="Duong dan file B (ghi de --b)")
    ap.add_argument("--round", type=int, default=None,
                    help="Vong IAA: doc qa/rounds/vqa_<a>_r<N>.csv vs vqa_<b>_r<N>.csv")
    args = ap.parse_args()

    if args.file_a:
        pa = Path(args.file_a)
    elif args.round:
        pa = QA_DIR / "rounds" / f"vqa_{args.a}_r{args.round}.csv"
    else:
        pa = QA_DIR / f"vqa_{args.a}.csv"
    if args.file_b:
        pb = Path(args.file_b)
    elif args.round:
        pb = QA_DIR / "rounds" / f"vqa_{args.b}_r{args.round}.csv"
    else:
        pb = QA_DIR / f"vqa_{args.b}.csv"
    for p in (pa, pb):
        if not p.exists():
            print(f"[LOI] khong thay file: {p}")
            return

    A = load(pa); B = load(pb)
    shared_files = A["files"] & B["files"]
    print("=" * 64)
    print(f"FILE A : {pa.name}   ({len(A['files'])} anh)")
    print(f"FILE B : {pb.name}   ({len(B['files'])} anh)")
    print(f"ANH GIAO NHAU (overlap, dung de tinh IAA): {len(shared_files)}")
    print("=" * 64)
    if not shared_files:
        print("Chua co anh nao ca 2 cung gan -> chua tinh duoc IAA.")
        return

    # ---------- TANG 2: dong thuan DAP AN ----------
    # gom cap dap an theo qid (chi tren anh giao nhau, va ca 2 cung hoi qid do)
    by_qid = defaultdict(list)   # qid -> list[(ansA, ansB)]
    qtype  = {}
    for (fn, qid), ra in A["rows"].items():
        if fn not in shared_files:
            continue
        rb = B["rows"].get((fn, qid))
        if rb is None:
            continue
        by_qid[qid].append((norm(ra["answer"]), norm(rb["answer"])))
        qtype[qid] = ra.get("question_type", "")

    print("\n[TANG 2] DONG THUAN DAP AN (cung anh + cung cau):")
    print(f"{'qid':<5}{'type':<10}{'n':>4}  {'%exact':>7}   kappa")
    print("-" * 60)
    yesno_pairs = []          # gop tat ca cau yes/no -> headline
    n_all = exact_all = 0
    for qid in sorted(by_qid):
        pairs = by_qid[qid]
        n = len(pairs)
        nexact = sum(1 for a, b in pairs if a == b)
        k = cohen_kappa(pairs)
        print(f"{qid:<5}{qtype.get(qid,''):<10}{n:>4}  {nexact/n*100:>6.1f}%   {kappa_str(k)}")
        n_all += n; exact_all += nexact
        if qtype.get(qid, "") == "yes_no":
            yesno_pairs.extend(pairs)

    print("-" * 60)
    if yesno_pairs:
        ky = cohen_kappa(yesno_pairs)
        exy = sum(1 for a, b in yesno_pairs if a == b) / len(yesno_pairs)
        print(f"GOP tat ca cau YES/NO : n={len(yesno_pairs)}  "
              f"%exact={exy*100:.1f}%  kappa={kappa_str(ky)}")
    if n_all:
        print(f"GOP tat ca cau        : n={n_all}  %exact={exact_all/n_all*100:.1f}%")

    # ---------- TANG 1: dong thuan CHON CAU ----------
    # (chi co y nghia khi 2 nguoi CHON CAU DOC LAP; voi quy trinh chot-chung thi
    #  selection trung nhau -> kappa thuong n/a. In ra de tham khao.)
    print("\n[TANG 1] DONG THUAN CHON CAU (tren anh giao nhau):")
    print(f"{'qid':<5}{'kappa(chon/khong)':<26}")
    print("-" * 40)
    all_opt_qids = set()
    for d in (A, B):
        for fn in shared_files:
            all_opt_qids |= d["sel"].get(fn, set())
    sel_kappas = []
    for qid in sorted(all_opt_qids):
        pr = []
        for fn in shared_files:
            pr.append((qid in A["sel"].get(fn, set()), qid in B["sel"].get(fn, set())))
        k = cohen_kappa(pr)
        if k is not None:
            sel_kappas.append(k)
        print(f"{qid:<5}{kappa_str(k)}")
    if sel_kappas:
        print("-" * 40)
        print(f"Selection kappa macro-avg: {sum(sel_kappas)/len(sel_kappas):+.3f}")

    print("\nNguong tham khao (guideline): kappa >= 0.7 / agreement >= 80% = dat.")
    print("=" * 64)


if __name__ == "__main__":
    main()

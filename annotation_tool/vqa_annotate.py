#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vqa_annotate.py — Gan nhan VQA thu cong bang NGAN HANG CAU HOI (10 loai, v2 Round 3)
Vietnamese Sidewalk VQA Dataset

Cap nhat v2 Round 3 (theo guideline_v2_round3.md):
  - 10 LOAI (Q1..Q10). Bo Q11. Q8 gop nhan mai che/o du; doi ten thung rac.
  - Q8 = 8 nhan cu the + 'vat can khac' (chon -> BAT BUOC go mo ta, luu cot explanation).
  - Moi loai co NHIEU CACH HOI (paraphrase). Khi hien thi, chon 1 cach hoi theo
    hash(ten_anh + qid) -> CO DINH cho moi (anh, cau): 2 nguoi trong mode select/answer
    luon thay cung mot cau chu; nhung khac nhau giua cac anh -> da dang toan dataset.
  - Sua lai cach tra loi (akind) tung loai.

3 che do (--mode):
  free   : 1 nguoi nhin anh -> chon cau hoi -> tra loi luon (cho KHOI BULK, nhanh).
  select : 2 nguoi ngoi chung, CHON cau hoi cho moi anh (khong tra loi) -> ghi
           ke hoach qa/iaa_plan.csv  (cho BO IAA). Cau chu paraphrase duoc ghi kem.
  answer : moi nguoi DOC LAP tra loi dung danh sach cau trong iaa_plan.csv -> ghi
           qa/vqa_<annotator>.csv. Sau do chay compute_iaa.py de so khop.

Dung:
  python vqa_annotate.py --annotator dung --mode free
  python vqa_annotate.py --annotator team --mode select
  python vqa_annotate.py --annotator dung --mode answer
  python compute_iaa.py --a dung --b ban

Cot CSV (vqa_<ann>.csv): filename, qid, question, answer, question_type, task,
                         cat, explanation, annotator, timestamp
Plan (iaa_plan.csv)    : filename, qid, question
"""

from __future__ import annotations

import sys
import csv
import ctypes
import hashlib
import argparse
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    from PIL import Image, ImageTk
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image, ImageTk

# ── Paths ──────────────────────────────────────────────────────────────────────
IMAGES_DIR = Path("./images_v2")   # 5055 anh da loc tay
QA_DIR     = Path("./qa")
PLAN_PATH  = QA_DIR / "iaa_plan.csv"
MAX_IMG_W, MAX_IMG_H = 720, 560
SKIP_QID   = "__skip__"

# Giai doan hien tai: GAN NHAN bang CAU HOI GOC (variants[0]) cho moi anh.
# Khi gan nhan xong, doi True de bat paraphrase (moi (anh,cau) chon 1 cach hoi
# theo hash(filename+qid) -> da dang dien dat toan dataset).
USE_PARAPHRASE = False

# ── Nhan chuan loai vi pham (theo thu tu CAT) — giu lai cho tham chieu ──────────
STD_LABELS = [
    ("CAT-1", "kinh doanh / biển hiệu trên vỉa hè"),
    ("CAT-2", "vật liệu xây dựng / máy móc trên vỉa hè"),
    ("CAT-3", "để xe trái phép"),
    ("CAT-4", "xe đi trên vỉa hè"),
    ("CAT-5", "vật cản / cấu trúc chiếm dụng vỉa hè"),
    ("CAT-6", "hư hỏng hạ tầng"),
]

# ── NGAN HANG CAU HOI (11 loai, v2 Round 1) ────────────────────────────────────
# akind   : binary | multichoice | choice | count | text
# variants: danh sach cach hoi (paraphrase). variants[0] = ban goc trong guideline.
#           Khi hien thi se chon 1 cach theo hash(filename+qid) -> on dinh moi (anh,cau).
# expl    : dict{dap_an: giai_thich}  hoac  str  (goi y, sua duoc; bam luat that)
QUESTION_BANK = [
    dict(id="Q1", section="Tổng quát", task="reasoning", qtype="yes_no", cat="", akind="binary",
         variants=[
            "Vỉa hè trong ảnh có coi là bị lấn chiếm/hư hỏng hoặc vi phạm về quy định sử dụng không?",
            "Trong ảnh, vỉa hè có bị chiếm dụng trái phép không?",
            "Có hành vi vi phạm quy định sử dụng vỉa hè nào xuất hiện trong ảnh không?",
            "Vỉa hè ở ảnh này có dấu hiệu bị lấn chiếm không?",
            "Ảnh này có cho thấy vỉa hè bị sử dụng sai quy định không?",
            "Phần vỉa hè trong ảnh có đang bị lấn chiếm không?",
         ],
         expl={"có": "vỉa hè bị chiếm dụng ngoài mục đích đi bộ mà không có giấy phép (NĐ 168/2024)",
               "không": "vỉa hè trống hoặc chỉ có hạ tầng hợp pháp"}),

    dict(id="Q2", section="Kinh doanh / biển hiệu", task="perception", qtype="what", cat="CAT-1", akind="multichoice",
         options=["quán ăn / quán nước", "hàng rong / sạp bán", "xe đẩy bán hàng",
                  "sửa xe / rửa xe", "cắt tóc / làm đẹp",
                  "cửa hàng bày hàng ra hè", "trưng bày xe để bán", "xổ số", "không có"],
         variants=[
            "Loại hình kinh doanh nào sau đây đang lấn chiếm vỉa hè trong ảnh?",
            "Hoạt động kinh doanh nào sau đây đang chiếm dụng vỉa hè trong ảnh?",
            "Vỉa hè trong ảnh bị lấn chiếm bởi loại hình buôn bán nào sau đây?",
            "Trong ảnh, hình thức kinh doanh nào sau đây đang lấn chiếm vỉa hè?",
            "Loại buôn bán/dịch vụ nào sau đây đang bày ra vỉa hè trong ảnh?",
            "Vỉa hè bị chiếm dụng để kinh doanh dưới hình thức nào sau đây trong ảnh?",
         ],
         expl="loại hoạt động kinh doanh chiếm dụng vỉa hè (NĐ 168/2024 Điều 12 khoản 7)"),

    dict(id="Q3", section="Kinh doanh / biển hiệu", task="reasoning", qtype="yes_no", cat="CAT-1", akind="binary",
         variants=[
            "Có biển hiệu/biển quảng cáo nào trong ảnh bị coi là lấn chiếm vỉa hè không?",
            "Trong ảnh có biển hiệu hoặc biển quảng cáo nào lấn chiếm vỉa hè không?",
            "Có tấm biển quảng cáo/biển hiệu nào đặt chiếm vỉa hè trong ảnh không?",
            "Ảnh này có biển hiệu, biển quảng cáo lấn ra vỉa hè không?",
            "Vỉa hè trong ảnh có bị biển hiệu/quảng cáo chiếm dụng không?",
            "Có biển quảng cáo hay biển hiệu nào vi phạm lấn chiếm vỉa hè trong ảnh không?",
         ],
         expl={"có": "biển cắm/dựng trên mặt vỉa hè, hoặc treo lên cây xanh/cột đèn/hạ tầng hè là chiếm dụng trái phép (NĐ 168/2024 Điều 12 khoản 7)",
               "không": "biển gắn trên tường nhà / treo trên cao từ mặt tiền, không chạm bề mặt hè và không treo vào hạ tầng hè"}),

    dict(id="Q4", section="Vật liệu / máy móc", task="perception", qtype="what", cat="CAT-2", akind="multichoice",
         options=["cát", "gạch", "đá", "xi măng", "sắt thép", "gỗ", "ống nước",
                  "máy xúc", "máy trộn bê tông", "xe lu", "giàn giáo",
                  "thiết bị/vật liệu", "không có"],
         variants=[
            "Loại vật liệu/thiết bị/vật tư hoặc máy móc nào sau đây đang lấn chiếm vỉa hè trong ảnh?",
            "Loại vật liệu xây dựng hoặc máy móc nào sau đây đang chiếm dụng vỉa hè trong ảnh?",
            "Trong ảnh, vỉa hè bị lấn chiếm bởi loại vật liệu/thiết bị/máy móc nào sau đây?",
            "Vật tư, vật liệu hay máy móc loại nào sau đây đang để trên vỉa hè trong ảnh?",
            "Có loại vật liệu/thiết bị/máy móc nào sau đây đang tập kết trên vỉa hè trong ảnh?",
            "Vỉa hè trong ảnh đang bị chiếm bởi loại vật liệu hoặc máy móc nào sau đây?",
         ],
         expl="vật liệu / máy móc / thiết bị / vật tư tập kết trên vỉa hè (NĐ 168/2024 Điều 12 khoản 9)"),

    dict(id="Q5", section="Đỗ xe", task="perception", qtype="what", cat="CAT-3", akind="multichoice",
         options=["xe máy", "ô tô", "xe đạp", "xe đạp điện", "xe ba gác", "xe tải", "không có"],
         variants=[
            "Loại phương tiện nào sau đây đang đỗ trái phép trên vỉa hè trong ảnh?",
            "Những loại xe nào sau đây đang đỗ trái phép trên vỉa hè trong ảnh?",
            "Trong ảnh, phương tiện loại nào sau đây đang đậu sai quy định trên vỉa hè?",
            "Vỉa hè trong ảnh bị loại phương tiện nào sau đây đỗ chiếm dụng?",
            "Có những loại xe nào sau đây đang đỗ trái phép trên vỉa hè trong ảnh?",
            "Phương tiện đỗ trên vỉa hè trong ảnh gồm những loại nào sau đây?",
         ],
         expl="loại phương tiện đỗ trái phép trên vỉa hè (NĐ 168/2024 Điều 7 kh.2đ / Điều 6 kh.2đ)"),

    dict(id="Q6", section="Đỗ xe", task="reasoning", qtype="how_many", cat="CAT-3", akind="count",
         variants=[
            "Trong ảnh có bao nhiêu phương tiện bị coi là đỗ trái phép trên vỉa hè?",
            "Có bao nhiêu xe đang đỗ trái phép trên vỉa hè trong ảnh?",
            "Trong ảnh, số phương tiện đỗ sai quy định trên vỉa hè là bao nhiêu?",
            "Đếm số phương tiện đỗ trái phép trên vỉa hè trong ảnh?",
            "Số lượng xe đỗ trái phép trên vỉa hè trong ảnh là bao nhiêu?",
            "Vỉa hè trong ảnh có mấy phương tiện đỗ trái phép?",
         ],
         expl="đếm phương tiện đỗ trái phép (loại trừ xe cắt ngang vào nhà, xe lăn, xe đang sửa ở tiệm); không có → 0"),

    dict(id="Q7", section="Xe di chuyển trên vỉa hè", task="reasoning", qtype="yes_no", cat="CAT-4", akind="binary",
         variants=[
            "Có xe nào đang di chuyển vi phạm trên vỉa hè trong ảnh không?",
            "Trong ảnh có xe đang chạy dọc trên vỉa hè (vi phạm) không?",
            "Có phương tiện nào đang di chuyển trái phép dọc vỉa hè trong ảnh không?",
            "Ảnh này có xe chạy hoặc dắt đi trên vỉa hè vi phạm không?",
            "Vỉa hè trong ảnh có xe nào đang lưu thông trái phép không?",
            "Có xe đang đi dọc trên vỉa hè (không phải cắt ngang vào nhà) trong ảnh không?",
         ],
         expl={"có": "xe chạy dọc / dắt đi tới trên hè là vi phạm (NĐ 168/2024 Điều 7 kh.7a)",
               "không": "không có xe di chuyển dọc hè; hoặc chỉ cắt ngang vào nhà / xe lăn (ngoại lệ hợp lệ)"}),

    dict(id="Q8", section="Vật cản / cấu trúc", task="perception", qtype="what", cat="CAT-5", akind="multichoice",
         options=["chậu cây cá nhân", "thùng rác/bọc rác cá nhân", "bục/bậc tự xây",
                  "mái che/ô dù", "ghế/bàn cá nhân",
                  "không có"],
         variants=[
            "Vỉa hè trong ảnh có loại vật cản/cấu trúc nào sau đây đang lấn chiếm?",
            "Vỉa hè trong ảnh đang bị loại vật cản/cấu trúc nào sau đây lấn chiếm?",
            "Có những loại vật cản hoặc cấu trúc nào sau đây chiếm dụng vỉa hè trong ảnh?",
            "Trong ảnh, vỉa hè bị chiếm bởi loại vật cản/cấu trúc nào sau đây?",
            "Loại chướng ngại vật hay cấu trúc nào sau đây đang lấn chiếm vỉa hè trong ảnh?",
            "Vỉa hè trong ảnh xuất hiện loại vật cản/cấu trúc nào sau đây?",
         ],
         expl="vật cản / cấu trúc cá nhân chiếm dụng vỉa hè (NĐ 168/2024 Điều 12 khoản 11a)"),

    dict(id="Q9", section="Hư hỏng hạ tầng", task="reasoning", qtype="yes_no", cat="CAT-6", akind="binary",
         variants=[
            "Bề mặt vỉa hè có bị bong tróc/sụt lún hoặc nắp cống hỏng/mất tạo hố, ổ gà, ổ voi gây nguy hiểm cho người đi bộ (kể cả người khuyết tật đi xe lăn) không?",
            "Mặt vỉa hè trong ảnh có bị hư hỏng (bong tróc, sụt lún, mất nắp cống, ổ gà) gây nguy hiểm cho người đi bộ không?",
            "Trong ảnh, vỉa hè có chỗ vỡ/lún hoặc nắp cống hỏng tạo hố nguy hiểm cho người đi bộ và xe lăn không?",
            "Bề mặt vỉa hè trong ảnh có hư hại đến mức gây nguy hiểm cho người đi bộ không?",
            "Có vị trí nào trên vỉa hè bị hỏng (ổ gà, hố, mất nắp cống) gây nguy cơ vấp ngã cho người đi bộ trong ảnh không?",
            "Vỉa hè trong ảnh có bị xuống cấp tạo hố/ổ gà nguy hiểm cho người đi bộ và xe lăn không?",
         ],
         expl={"có": "hư hỏng tạo hố / mảnh nhọn / mất nắp cống trên lối đi gây nguy cơ vấp ngã (kể cả chặn xe lăn) — Luật 36/2024 Điều 77",
               "không": "mặt hè còn nguyên hoặc chỉ bong tróc bề mặt không tạo hố, không nguy hiểm"}),

    dict(id="Q10", section="Lối đi bộ", task="reasoning", qtype="spatial", cat="", akind="binary",
         variants=[
            "Người đi bộ còn lối đi thông suốt trên vỉa hè trong ảnh không?",
            "Trong ảnh, người đi bộ có còn lối đi thông suốt trên vỉa hè không?",
            "Vỉa hè trong ảnh có còn đủ lối cho người đi bộ đi qua không?",
            "Người đi bộ trong ảnh có còn đường đi liên tục trên vỉa hè không?",
            "Lối đi bộ trên vỉa hè trong ảnh có còn thông suốt không?",
            "Trong ảnh, người đi bộ có thể đi trên vỉa hè mà không phải xuống lòng đường không?",
         ],
         expl={"có": "còn một dải hè liên tục cho người đi bộ đi qua (kể cả phải lách)",
               "không": "vật cản/xe chặn gần hết bề rộng hè → người đi bộ buộc phải xuống lòng đường (tham chiếu NĐ 165/2024 Điều 21 — chừa 1,5m)"}),

]

QBANK = {q["id"]: q for q in QUESTION_BANK}
# gan text mac dinh = cach hoi goc (dung cho status bar / fallback)
for _q in QUESTION_BANK:
    _q["text"] = _q["variants"][0]

FIELDNAMES = ["filename", "qid", "question", "answer", "question_type",
              "task", "cat", "explanation", "annotator", "timestamp"]

# ── HUONG DAN TRA LOI tung cau (trich guideline_round_4.md Phan C) ──────────────
ANSWER_GUIDE = {
    "Q1": ("• có: xuất hiện ÍT NHẤT MỘT vi phạm (kinh doanh, vật liệu/máy móc, đỗ xe, xe chạy trên hè, "
           "vật cản/cấu trúc, hư hỏng nguy hiểm).\n"
           "• không: hè trống hoặc CHỈ có hạ tầng hợp pháp (cây đô thị, cột điện, biển báo, "
           "thùng rác/ghế công cộng, quốc kỳ).\n"
           "• Rác vụn lẻ tẻ (1 vỏ chai, túi nhỏ, cục gạch) → bỏ qua, vẫn 'không'."),
    "Q2": ("• Chọn TẤT CẢ loại kinh doanh chiếm hè (tự nối 'và'). KHÔNG có → 'không có'.\n"
           "• Tính KỂ CẢ không có người (bày hàng/bố trí bàn ghế = vi phạm).\n"
           "• Dịch vụ nhỏ (sửa đồng hồ, chìa khóa, giày...) dùng bàn/tủ nhỏ → 'xe đẩy bán hàng'.\n"
           "• Tiệm sửa/rửa xe → 'sửa xe / rửa xe' (KHÔNG tính sang đỗ xe). Không rõ bán gì → "
           "'cửa hàng bày hàng ra hè' hoặc bỏ câu.\n"
           "• Phân biệt: bàn ghế khách NGỒI = quán ăn/nước · xe đẩy/thiết bị/quầy = xe đẩy · "
           "sạp/mẹt bán mang đi = hàng rong/sạp bán."),
    "Q3": ("• có: biển CẮM/DỰNG trên mặt hè (chân biển chạm hè, standee, bảng chữ A); "
           "hoặc biển TREO lên cây xanh đô thị/cột đèn/hạ tầng hè.\n"
           "• không: biển GẮN TƯỜNG nhà / treo trên cao từ mặt tiền, không chạm/không gắn vào hạ tầng hè.\n"
           "• Không phân biệt được vị trí → bỏ câu."),
    "Q4": ("• Chọn TẤT CẢ loại vật liệu/máy móc (tự nối 'và'). KHÔNG có → 'không có'.\n"
           "• Cát/gạch/đá/xi măng/sắt thép/gỗ/ống nước; máy xúc/máy trộn/xe lu/giàn giáo.\n"
           "• Không rõ là vật liệu hay thiết bị gì cụ thể → 'thiết bị/vật liệu'.\n"
           "• Hàng TIÊU DÙNG bán lẻ → KHÔNG phải đây (thuộc Q2). Đồ sinh hoạt chắn lối → Q8.\n"
           "• Có biển/rào công trình (có thể có phép) → bỏ câu."),
    "Q5": ("• Chọn TẤT CẢ loại xe ĐỨNG YÊN trên hè (tự nối 'và'). KHÔNG có → 'không có'.\n"
           "• LOẠI TRỪ (→ 'không có' nếu chỉ có): xe cắt ngang vào nhà, xe lăn, xe đang sửa ở tiệm "
           "(thuộc Q2), xe đang di chuyển (thuộc Q7).\n"
           "• Người ngồi trên xe chân chạm đất, xe đứng yên → vẫn là đỗ. Không rõ đỗ hay chạy → bỏ câu."),
    "Q6": ("• Đếm từng xe rời đỗ trái phép. KHÔNG có xe đỗ → 0.\n"
           "• LOẠI TRỪ: xe cắt ngang vào nhà, xe lăn, xe đang sửa ở tiệm, xe đang di chuyển.\n"
           "• Quá nhiều / bị che, đếm không chắc → bỏ câu. Phải khớp Q5 ('không có' ⇒ 0)."),
    "Q7": ("• có: xe đang CHẠY DỌC trên hè (motion blur, bánh nghiêng đang lăn, tư thế đang lái) "
           "hoặc người ĐANG DẮT xe đi tới dọc hè.\n"
           "• không: không có xe di chuyển dọc hè — hè trống; chỉ có xe đứng yên (→ Q5); "
           "xe cắt NGANG vào cổng/nhà; xe lăn người khuyết tật.\n"
           "• Ảnh tĩnh không phân biệt được đứng yên/đang chạy → bỏ câu."),
    "Q8": ("• Chọn TẤT CẢ loại vật cản (tự nối 'và'). KHÔNG có → 'không có'.\n"
           "• 5 nhãn: chậu cây cá nhân (≠ cây đô thị) · thùng rác/bọc rác cá nhân (≠ thùng công cộng) · "
           "bục/bậc tự xây · mái che/ô dù (cố định hoặc di động, không phục vụ kinh doanh) · "
           "ghế/bàn cá nhân (≠ ghế quán).\n"
           "• Vật KHÔNG thuộc 5 nhãn (hàng rào/cọc, bình/phuy/bồn, ván/tấm lát, tủ cũ...) → BỎ CÂU.\n"
           "• KHÔNG chắc là gì / không rõ vi phạm (che, mờ, không rõ cá nhân/công cộng) → BỎ CÂU."),
    "Q9": ("• có: gạch vỡ tạo mảnh nhọn/hố, ổ gà, ổ voi, nắp cống mất/lõm/hỏng, sụt lún — TRÊN LỐI ĐI, "
           "gây nguy cơ vấp ngã (kể cả chặn xe lăn).\n"
           "• không: mặt hè còn nguyên; hoặc chỉ bong tróc/nứt không tạo hố; hư hỏng ở gốc cây ngoài lối đi.\n"
           "• Phép thử: người đi bộ / xe lăn có thể vấp/kẹt/bị thương ở đây không?"),
    "Q10": ("• Đây là câu MÔ TẢ KHÔNG GIAN, độc lập với chuyện hè có vi phạm hay không.\n"
            "• có: còn một DẢI HÈ LIÊN TỤC cho người đi bộ đi qua (kể cả phải lách/đi hẹp).\n"
            "• không: vật cản/xe/hàng quán chặn GẦN HẾT bề rộng hè → người đi bộ BUỘC phải xuống lòng đường.\n"
            "• Hè trống → có. Không nhìn được toàn bộ bề rộng hè (góc hẹp, bị che) → bỏ câu."),
}


# ── Chon cach hoi (paraphrase) on dinh theo (filename, qid) ─────────────────────
def pick_variant(filename: str, q: dict) -> str:
    """Chon 1 cach hoi tu q['variants'] mot cach XAC DINH theo hash(filename+qid).
    Voi cau multichoice / choice: tu dong append danh sach dap an vao cuoi cau hoi
    -> cot 'question' trong CSV luu day du ca cau hoi lan lua chon."""
    variants = q.get("variants") or [q.get("text", "")]
    if not USE_PARAPHRASE or len(variants) <= 1:
        base = variants[0]
    else:
        key = f"{filename}|{q['id']}".encode("utf-8")
        idx = int(hashlib.md5(key).hexdigest(), 16) % len(variants)
        base = variants[idx]
    options = q.get("options")
    if options and q.get("akind") in ("multichoice", "choice"):
        base = base.rstrip("?") + "? (" + " · ".join(options) + ")"
    return base


# ── Windows: chan Alt+Space system-menu (xung dot Unikey IME) ──────────────────
_WNDPROC_REF = None

def _suppress_keymenu(root: tk.Tk) -> None:
    global _WNDPROC_REF
    try:
        from ctypes import wintypes
        WM_SYSCOMMAND = 0x0112
        SC_KEYMENU    = 0xF100
        GWLP_WNDPROC  = -4
        WNDPROC = ctypes.WINFUNCTYPE(
            ctypes.c_long, wintypes.HWND, wintypes.UINT,
            wintypes.WPARAM, wintypes.LPARAM,
        )
        u32 = ctypes.windll.user32
        CallWindowProc          = u32.CallWindowProcW
        CallWindowProc.restype  = ctypes.c_long
        CallWindowProc.argtypes = [ctypes.c_void_p, wintypes.HWND,
                                   wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        try:
            GetWndLong = u32.GetWindowLongPtrW
            SetWndLong = u32.SetWindowLongPtrW
        except AttributeError:
            GetWndLong = u32.GetWindowLongW
            SetWndLong = u32.SetWindowLongW
        GetWndLong.restype  = ctypes.c_void_p
        GetWndLong.argtypes = [wintypes.HWND, ctypes.c_int]
        SetWndLong.restype  = ctypes.c_void_p
        SetWndLong.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
        root.update_idletasks()
        wid  = root.winfo_id()
        hwnd = u32.GetParent(wid) or wid
        original = GetWndLong(hwnd, GWLP_WNDPROC)
        @WNDPROC
        def _proc(h, msg, wp, lp):
            if msg == WM_SYSCOMMAND and (wp & 0xFFF0) == SC_KEYMENU:
                return 0
            return CallWindowProc(original, h, msg, wp, lp)
        _WNDPROC_REF = _proc
        SetWndLong(hwnd, GWLP_WNDPROC, _proc)
    except Exception:
        pass


def _suggest_expl(q: dict, answer: str) -> str:
    expl = q.get("expl")
    if isinstance(expl, dict):
        return expl.get(answer, "")
    return expl or ""


# ── Dialog tra loi 1 cau hoi ───────────────────────────────────────────────────
class AnswerDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, q: dict, qtext: str | None = None):
        super().__init__(parent)
        self.q      = q
        self.qtext  = qtext or q["text"]      # cau chu paraphrase da chon cho anh nay
        self.result = None
        self.title("Trả lời câu hỏi")
        self.configure(bg="#252526")
        self.transient(parent)
        self.minsize(560, 320)

        tk.Label(self, text=self.qtext, bg="#252526", fg="#FFD700",
                 font=("Segoe UI", 11, "bold"), wraplength=520, justify="left"
                 ).pack(anchor="w", padx=14, pady=(12, 8))
        badge = f"[{q['qtype']}] [{q['task']}]" + (f" [{q['cat']}]" if q['cat'] else "")
        tk.Label(self, text=badge, bg="#252526", fg="#888",
                 font=("Consolas", 9)).pack(anchor="w", padx=14)

        guide = ANSWER_GUIDE.get(q["id"])
        if guide:
            gf = tk.Frame(self, bg="#16241a")
            gf.pack(fill="x", padx=14, pady=(8, 2))
            tk.Label(gf, text="HƯỚNG DẪN TRẢ LỜI (guideline v2 Round 1):", bg="#16241a", fg="#9ccc65",
                     font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=8, pady=(5, 0))
            tk.Label(gf, text=guide, bg="#16241a", fg="#dcedc8", font=("Segoe UI", 9),
                     wraplength=508, justify="left").pack(anchor="w", padx=8, pady=(1, 6))

        body = tk.Frame(self, bg="#252526"); body.pack(fill="x", padx=14, pady=10)
        self.akind   = q["akind"]
        self.ans_var = tk.StringVar(value="")
        self.viol_vars: dict[str, tk.BooleanVar] = {}

        if self.akind == "binary":
            self._radios(body, ["có", "không"])
        elif self.akind == "choice":
            self._radios(body, q["options"])
        elif self.akind == "multichoice":
            tk.Label(body, text="Chọn TẤT CẢ loại có trong ảnh (tự nối \"và\"); không có loại nào → \"không có\":",
                     bg="#252526", fg="#bbb", font=("Segoe UI", 9)).pack(anchor="w")
            self.multi_vars = {}
            for opt in q["options"]:
                v = tk.BooleanVar(value=False)
                self.multi_vars[opt] = v
                tk.Checkbutton(body, text=opt, variable=v,
                               bg="#252526", fg="#a5d6a7", selectcolor="#0d1117",
                               activebackground="#252526", activeforeground="#a5d6a7",
                               font=("Segoe UI", 11), anchor="w").pack(anchor="w", fill="x")
            self.multi_desc_var = tk.StringVar(value="")
            if q.get("desc_option"):
                tk.Label(body, text=f"Mô tả '{q['desc_option']}' (TÙY CHỌN — ghi tên vật nếu thấy nhanh):",
                         bg="#252526", fg="#ffcc80", font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))
                tk.Entry(body, textvariable=self.multi_desc_var, font=("Segoe UI", 11),
                         bg="#0d0d1a", fg="#ffe0b2", insertbackground="white").pack(fill="x", pady=2)
        elif self.akind == "violations":
            tk.Label(body, text="Chọn HẾT loại vi phạm có trong ảnh (không chọn = không vi phạm):",
                     bg="#252526", fg="#bbb", font=("Segoe UI", 9)).pack(anchor="w")
            for code, label in STD_LABELS:
                v = tk.BooleanVar(value=False)
                self.viol_vars[code] = v
                tk.Checkbutton(body, text=label, variable=v,
                               bg="#252526", fg="#a5d6a7", selectcolor="#0d1117",
                               activebackground="#252526", activeforeground="#a5d6a7",
                               font=("Segoe UI", 10), anchor="w").pack(anchor="w", fill="x")
        elif self.akind == "count":
            tk.Label(body, text="Số lượng:", bg="#252526", fg="#bbb",
                     font=("Segoe UI", 10)).pack(side="left")
            self.ans_var.set("1")
            tk.Spinbox(body, from_=0, to=50, textvariable=self.ans_var, width=6,
                       font=("Consolas", 12)).pack(side="left", padx=8)
        elif self.akind == "text":
            tk.Label(body, text="Nhập câu trả lời ngắn (danh từ / cụm danh từ):",
                     bg="#252526", fg="#bbb", font=("Segoe UI", 9)).pack(anchor="w")
            e = tk.Entry(body, textvariable=self.ans_var, font=("Segoe UI", 12),
                         bg="#0d0d1a", fg="#a5d6a7", insertbackground="white")
            e.pack(fill="x", pady=4); e.focus_set()
            sugg = q.get("suggest") or []
            if sugg:
                tk.Label(body, text="Gợi ý (bấm để điền; nhiều thứ → bấm tiếp, tự nối \"và\"):",
                         bg="#252526", fg="#888", font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))
                chips = tk.Frame(body, bg="#252526"); chips.pack(fill="x")
                cols = 3
                for i, s in enumerate(sugg):
                    tk.Button(chips, text=s, bg="#33334a", fg="#cfefff", relief="flat",
                              font=("Segoe UI", 9), padx=6, pady=2,
                              command=lambda ss=s: self._fill_text(ss)
                              ).grid(row=i // cols, column=i % cols, sticky="ew", padx=2, pady=2)
                for c in range(cols):
                    chips.grid_columnconfigure(c, weight=1)
                tk.Button(body, text="xóa ô nhập", bg="#3a2a2a", fg="#ffcdd2", relief="flat",
                          font=("Segoe UI", 8), command=lambda: self.ans_var.set("")
                          ).pack(anchor="w", pady=(2, 0))

        ef = tk.Frame(self, bg="#252526"); ef.pack(fill="x", padx=14, pady=(2, 6))
        tk.Label(ef, text="Giải thích (gợi ý theo luật — sửa/để trống tùy ý):",
                 bg="#252526", fg="#888", font=("Segoe UI", 8)).pack(anchor="w")
        init_expl = "" if isinstance(q.get("expl"), dict) else _suggest_expl(q, "")
        self.expl_var = tk.StringVar(value=init_expl)
        tk.Entry(ef, textvariable=self.expl_var, font=("Segoe UI", 9),
                 bg="#15151f", fg="#cfcfcf", insertbackground="white").pack(fill="x")

        bf = tk.Frame(self, bg="#252526"); bf.pack(fill="x", padx=14, pady=10)
        tk.Button(bf, text="Thêm câu này  (Enter)", bg="#2e7d32", fg="white",
                  font=("Segoe UI", 10, "bold"), relief="flat", padx=10, pady=6,
                  command=self._ok).pack(side="left")
        tk.Button(bf, text="Hủy (Esc)", bg="#5a1a1a", fg="#ffcdd2",
                  font=("Segoe UI", 9), relief="flat", padx=10, pady=6,
                  command=self._cancel).pack(side="left", padx=8)

        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self._cancel())
        self.grab_set()
        self.wait_window(self)

    def _radios(self, parent, options):
        for opt in options:
            tk.Radiobutton(parent, text=opt, value=opt, variable=self.ans_var,
                           bg="#252526", fg="#a5d6a7", selectcolor="#0d1117",
                           activebackground="#252526", activeforeground="#fff",
                           font=("Segoe UI", 12), anchor="w",
                           command=self._on_change).pack(anchor="w", fill="x", pady=2)

    def _fill_text(self, s: str):
        cur = self.ans_var.get().strip()
        self.ans_var.set(s if not cur else f"{cur} và {s}")

    def _on_change(self):
        if self.akind in ("binary", "choice"):
            ans = self.ans_var.get()
            if isinstance(self.q.get("expl"), dict) and ans:
                self.expl_var.set(self.q["expl"].get(ans, ""))

    def _ok(self):
        q = self.q
        cat = q["cat"]
        extra_desc = ""
        if self.akind == "violations":
            picked = [c for c, _ in STD_LABELS if self.viol_vars[c].get()]
            labels = [lab for c, lab in STD_LABELS if c in picked]
            answer = " + ".join(labels) if labels else "không vi phạm"
            cat    = ",".join(picked)
        elif self.akind == "multichoice":
            picked = [opt for opt in q["options"] if self.multi_vars[opt].get()]
            if not picked:
                messagebox.showwarning("Thiếu", "Chọn ít nhất 1 loại (hoặc 'không có').", parent=self); return
            if "không có" in picked and len(picked) > 1:
                messagebox.showwarning("Mâu thuẫn",
                    "Không thể chọn 'không có' cùng với loại khác.", parent=self); return
            desc_opt = q.get("desc_option")
            if desc_opt and desc_opt in picked:
                extra_desc = self.multi_desc_var.get().strip()   # TUY CHON, khong bat buoc
            answer = " và ".join(picked)        # thu tu co dinh theo options -> IAA khop
        elif self.akind == "count":
            answer = self.ans_var.get().strip()
            if not answer.isdigit():
                messagebox.showwarning("Thiếu", "Nhập một số.", parent=self); return
        else:
            answer = self.ans_var.get().strip()
            if not answer:
                messagebox.showwarning("Thiếu", "Chưa chọn/nhập câu trả lời.", parent=self); return
        expl_out = self.expl_var.get().strip()
        if extra_desc:
            tag = f"{q.get('desc_option')}: {extra_desc}"
            expl_out = tag if not expl_out else f"{tag} | {expl_out}"
        self.result = {
            "qid": q["id"], "question": self.qtext, "answer": answer,
            "question_type": q["qtype"], "task": q["task"], "cat": cat,
            "explanation": expl_out,
        }
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()


# ── App chinh ─────────────────────────────────────────────────────────────────
class AnnotateApp:
    def __init__(self, root, annotator, images_dir, reverse, mode, plan_arg, round_n=None):
        self.root       = root
        self.annotator  = annotator
        self.images_dir = images_dir
        self.mode       = mode
        self.round_n    = round_n
        # --plan co the la NHIEU file ngan cach dau phay (mode answer doc het);
        # mode select ghi vao file DAU TIEN.
        self.plan_paths = [Path(x.strip()) for x in str(plan_arg).split(",") if x.strip()]
        self.plan_path  = self.plan_paths[0]
        QA_DIR.mkdir(parents=True, exist_ok=True)
        if round_n:                              # moi round co file rieng + danh sach anh rieng
            rd = QA_DIR / "rounds"; rd.mkdir(parents=True, exist_ok=True)
            self.out_path = rd / f"vqa_{annotator}_r{round_n}.csv"
            self.rej_path = rd / f"rejected_{annotator}_r{round_n}.csv"
            self.images_list_file = rd / f"round_{round_n}.txt"
        else:
            self.out_path = QA_DIR / f"vqa_{annotator}.csv"
            self.rej_path = QA_DIR / f"rejected_vqa_{annotator}.csv"
            self.images_list_file = None

        self.plan: dict[str, list[str]] = {}
        if mode == "answer":
            self.plan = self._load_plan()
            if not self.plan:
                messagebox.showerror("Lỗi", f"Chưa có kế hoạch câu hỏi:\n{self.plan_path}\n"
                                            "Hãy chạy --mode select trước.")
                root.destroy(); return
            base = sorted(self.plan.keys())
        elif self.images_list_file is not None:
            if not self.images_list_file.exists():
                messagebox.showerror("Lỗi", f"Chưa có danh sách ảnh của round:\n{self.images_list_file}")
                root.destroy(); return
            existing = {p.name for ext in ("*.jpg", "*.jpeg", "*.png") for p in images_dir.glob(ext)}
            base = [ln.strip() for ln in self.images_list_file.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and ln.strip() in existing]
        else:
            base = sorted(p.name for ext in ("*.jpg", "*.jpeg", "*.png")
                          for p in images_dir.glob(ext))
        if reverse:
            base = list(reversed(base))
        if not base:
            messagebox.showerror("Lỗi", "Không có ảnh để làm.")
            root.destroy(); return

        done = self._load_done()
        self.files = [f for f in base if f not in done]
        self.n_done_prev = len(done)
        if not self.files:
            messagebox.showinfo("Xong", "Tất cả ảnh trong phạm vi đã được làm!")
            root.destroy(); return

        self.idx = 0
        self.cur_pairs: list[dict] = []   # free/answer: da tra loi; select: {qid,question}
        self._build_ui()
        _suppress_keymenu(root)
        self._load_image()

    # ---- IO helpers ----
    def _load_plan(self) -> dict[str, list[str]]:
        plan: dict[str, list[str]] = defaultdict(list)
        for pp in self.plan_paths:          # gop nhieu file plan, khu trung lap qid
            if not pp.exists():
                continue
            with open(pp, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    qid = row.get("qid", "")
                    fn  = row.get("filename", "")
                    if qid and fn and qid != SKIP_QID and qid not in plan[fn]:
                        plan[fn].append(qid)
        return {k: v for k, v in plan.items() if v}

    def _load_done(self) -> set[str]:
        done: set[str] = set()
        if self.mode == "select":
            if self.plan_path.exists():
                with open(self.plan_path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row.get("filename"):
                            done.add(row["filename"])
        else:  # free / answer -> da tra loi (hoac loai)
            for p in (self.out_path, self.rej_path):
                if p.exists():
                    with open(p, newline="", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            if row.get("filename"):
                                done.add(row["filename"])
        return done

    # ---- helper: cau chu paraphrase cho anh hien tai ----
    def _qtext(self, q: dict) -> str:
        return pick_variant(self.files[self.idx], q)

    # ---- UI ----
    def _build_ui(self):
        self.root.title(f"VQA Annotate [{self.mode}] — {self.annotator}")
        self.root.configure(bg="#1e1e1e")
        self.root.geometry("1360x880")

        top = tk.Frame(self.root, bg="#2d2d2d"); top.pack(fill="x")
        self.status_var = tk.StringVar()
        tk.Label(top, textvariable=self.status_var, bg="#2d2d2d", fg="white",
                 font=("Consolas", 10), anchor="w", padx=10, pady=5).pack(side="left")
        modecol = {"free": "#1565C0", "select": "#6a1b9a", "answer": "#00695c"}[self.mode]
        tk.Label(top, text=f" {self.mode} · {self.annotator} ", bg=modecol, fg="white",
                 font=("Consolas", 10, "bold"), padx=8, pady=5).pack(side="right", padx=4)

        main = tk.Frame(self.root, bg="#1e1e1e"); main.pack(fill="both", expand=True, padx=4, pady=4)
        left = tk.Frame(main, bg="#121212"); left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, bg="#121212", highlightthickness=0,
                                width=MAX_IMG_W, height=MAX_IMG_H)
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)

        right = tk.Frame(main, bg="#252526", width=560); right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._build_right(right)

        self.root.bind("<Control-s>",     lambda e: self._save_next())
        self.root.bind("<Control-Right>", lambda e: self._save_next())
        self.root.bind("<Control-Left>",  lambda e: self._prev())
        self.root.bind("<Delete>",        lambda e: self._reject())
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def _build_right(self, parent):
        self.added_head = {"free": "CÂU ĐÃ TRẢ LỜI", "select": "CÂU ĐÃ CHỌN",
                           "answer": "CÂU ĐÃ TRẢ LỜI"}[self.mode]
        self.added_visible = False                       # mac dinh AN cho gon
        self.added_hdr = tk.Button(parent, anchor="w", relief="flat", bg="#252526",
                                   fg="#FFD700", activebackground="#333", activeforeground="#FFD700",
                                   font=("Consolas", 10, "bold"), command=self._toggle_added)
        self.added_hdr.pack(fill="x", padx=8, pady=(6, 2))
        self.added_frame = tk.Frame(parent, bg="#1b1b1b")
        # khong pack -> dang an; bam tieu de moi hien

        nav = tk.Frame(parent, bg="#252526"); nav.pack(fill="x", padx=8, pady=6)
        savetxt = "[Ctrl+S] Lưu kế hoạch & tiếp →" if self.mode == "select" else "[Ctrl+S] Lưu & tiếp →"
        tk.Button(nav, text=savetxt, bg="#2e7d32", fg="white",
                  font=("Consolas", 10, "bold"), relief="flat", padx=8, pady=6,
                  command=self._save_next).pack(side="left")
        tk.Button(nav, text="← trước", bg="#37474f", fg="white", font=("Consolas", 9),
                  relief="flat", padx=8, pady=6, command=self._prev).pack(side="left", padx=6)
        if self.mode != "answer":
            tk.Button(nav, text="[Del] Loại/bỏ ảnh", bg="#5a1a1a", fg="#ffcdd2",
                      font=("Consolas", 9), relief="flat", padx=8, pady=6,
                      command=self._reject).pack(side="right")

        banktitle = {"free": "NGÂN HÀNG CÂU HỎI  (bấm để trả lời)",
                     "select": "NGÂN HÀNG CÂU HỎI  (bấm CHỌN / bấm lại BỎ; xanh = đã chọn)",
                     "answer": "CÂU HỎI ĐÃ CHỐT CHO ẢNH NÀY  (bấm để trả lời)"}[self.mode]
        tk.Label(parent, text=banktitle, bg="#252526", fg="#4fc3f7",
                 font=("Consolas", 10, "bold")).pack(anchor="w", padx=8, pady=(4, 2))

        wrap = tk.Frame(parent, bg="#252526"); wrap.pack(fill="both", expand=True, padx=4, pady=2)
        self.bank_canvas = tk.Canvas(wrap, bg="#252526", highlightthickness=0)
        vsb = tk.Scrollbar(wrap, orient="vertical", command=self.bank_canvas.yview)
        self.bank_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.bank_canvas.pack(side="left", fill="both", expand=True)
        self.bank_inner = tk.Frame(self.bank_canvas, bg="#252526")
        self.bank_canvas.create_window((0, 0), window=self.bank_inner, anchor="nw")
        self.bank_inner.bind("<Configure>",
                             lambda e: self.bank_canvas.configure(scrollregion=self.bank_canvas.bbox("all")))
        self.bank_canvas.bind_all("<MouseWheel>",
                                  lambda e: self.bank_canvas.yview_scroll(int(-e.delta / 120), "units"))

    def _populate_bank(self):
        for w in self.bank_inner.winfo_children():
            w.destroy()
        self.q_buttons = {}
        if self.mode == "answer":
            qids = self.plan.get(self.files[self.idx], [])
            qlist = [QBANK[q] for q in qids if q in QBANK]
        else:
            qlist = QUESTION_BANK
        last = None
        for i, q in enumerate(qlist, 1):
            if q["section"] != last:
                tk.Label(self.bank_inner, text=q["section"], bg="#252526", fg="#ffb74d",
                         font=("Consolas", 9, "bold")).pack(anchor="w", padx=4, pady=(8, 1))
                last = q["section"]
            btn = tk.Button(self.bank_inner, text=f"{i}. {self._qtext(q)}", bg="#2a2a3a", fg="#ddd",
                            font=("Segoe UI", 9), relief="flat", anchor="w",
                            wraplength=470, justify="left", padx=8, pady=4,
                            command=lambda qq=q: self._on_question(qq))
            btn.pack(fill="x", padx=4, pady=1)
            self.q_buttons[q["id"]] = btn
        self._sync_buttons()

    def _sync_buttons(self):
        """To xanh nut cua cau da chon/da tra loi; con lai mau mac dinh."""
        present = {p["qid"] for p in self.cur_pairs}
        for qid, btn in getattr(self, "q_buttons", {}).items():
            if qid in present:
                btn.config(bg="#2e7d32", fg="white")
            else:
                btn.config(bg="#2a2a3a", fg="#ddd")

    def _toggle_added(self):
        self.added_visible = not self.added_visible
        if self.added_visible:
            self.added_frame.pack(fill="x", padx=8, after=self.added_hdr)
        else:
            self.added_frame.pack_forget()
        self._update_added_hdr()

    def _update_added_hdr(self):
        n = len(self.cur_pairs)
        arrow = "▼ ẩn bớt" if self.added_visible else "▶ bấm để xem"
        self.added_hdr.config(text=f"  {self.added_head}: {n}    [{arrow}]")

    def _refresh_added(self):
        self._update_added_hdr()
        for w in self.added_frame.winfo_children():
            w.destroy()
        if not self.cur_pairs:
            tk.Label(self.added_frame, text="(chưa có — bấm câu hỏi bên dưới)",
                     bg="#1b1b1b", fg="#666", font=("Segoe UI", 9)).pack(anchor="w", padx=6, pady=4)
            return
        for i, p in enumerate(self.cur_pairs):
            row = tk.Frame(self.added_frame, bg="#1b1b1b"); row.pack(fill="x", pady=1)
            tk.Button(row, text="✕", bg="#5a1a1a", fg="white", font=("Consolas", 8),
                      relief="flat", padx=4, command=lambda k=i: self._del_pair(k)).pack(side="right")
            if "answer" in p:
                tk.Label(row, text=f"[{p['answer']}]", bg="#1b1b1b", fg="#a5d6a7",
                         font=("Consolas", 9, "bold")).pack(side="right", padx=4)
            tk.Label(row, text=f"{p['qid']} · {p['question']}", bg="#1b1b1b", fg="#ccc",
                     font=("Segoe UI", 9), wraplength=380, justify="left", anchor="w").pack(side="left", padx=4)

    def _on_question(self, q: dict):
        qid = q["id"]
        qtext = self._qtext(q)
        present = any(p["qid"] == qid for p in self.cur_pairs)
        if self.mode == "select":
            if present:                              # bam lai -> bo chon
                self.cur_pairs = [p for p in self.cur_pairs if p["qid"] != qid]
            else:                                    # chon
                self.cur_pairs.append({"qid": qid, "question": qtext})
            self._refresh_added(); self._sync_buttons()
            return
        # free / answer
        if present and not messagebox.askyesno("Trùng", f"Câu {qid} đã thêm. Thêm nữa?"):
            return
        dlg = AnswerDialog(self.root, q, qtext)
        if dlg.result:
            self.cur_pairs.append(dlg.result)
            self._refresh_added(); self._sync_buttons()

    def _del_pair(self, k: int):
        if 0 <= k < len(self.cur_pairs):
            self.cur_pairs.pop(k); self._refresh_added(); self._sync_buttons()

    # ---- nav / save ----
    def _load_image(self):
        fn = self.files[self.idx]
        self.cur_pairs = []
        self._populate_bank()
        self._refresh_added()
        extra = f"  |  cần trả lời {len(self.plan.get(fn, []))} câu đã chốt" if self.mode == "answer" else ""
        self.status_var.set(f"  [{self.idx+1}/{len(self.files)}]  {fn}"
                            f"   |  đã xong: {self.n_done_prev}{extra}")
        try:
            img = Image.open(self.images_dir / fn)
            img.thumbnail((MAX_IMG_W, MAX_IMG_H), Image.LANCZOS)
            self._tk_img = ImageTk.PhotoImage(img)
            self.canvas.config(width=img.width, height=img.height)
            self.canvas.delete("all")
            self.canvas.create_image(img.width // 2, img.height // 2, anchor="center", image=self._tk_img)
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(360, 270, text=f"Không mở được ảnh:\n{e}",
                                    fill="red", font=("Consolas", 12))

    def _append(self, path: Path, fieldnames: list[str], rows: list[dict]):
        write_header = (not path.exists() or path.stat().st_size == 0)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerows(rows)

    def _save_next(self):
        fn = self.files[self.idx]
        answered = {p["qid"] for p in self.cur_pairs}

        if self.mode == "select":
            if not self.cur_pairs:
                messagebox.showwarning("Trống",
                    "Chưa chọn câu nào cho ảnh này.\n"
                    "Chọn ít nhất 1 câu, hoặc [Del] để bỏ ảnh khỏi bộ IAA.")
                return
            rows = [{"filename": fn, "qid": p["qid"], "question": p["question"]}
                    for p in self.cur_pairs]
            self._append(self.plan_path, ["filename", "qid", "question"], rows)

        elif self.mode == "answer":
            need = [q for q in self.plan.get(fn, []) if q not in answered]
            if need:
                messagebox.showwarning("Chưa trả lời hết",
                    "Phải trả lời HẾT các câu đã chốt cho ảnh này.\n"
                    f"Còn thiếu: {', '.join(need)}")
                return
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rows = [{"filename": fn, **p, "annotator": self.annotator, "timestamp": ts}
                    for p in self.cur_pairs]
            self._append(self.out_path, FIELDNAMES, rows)

        else:  # free
            if not self.cur_pairs:
                if not messagebox.askyesno("Trống", "Chưa thêm câu nào. Bỏ qua ảnh (không lưu)?"):
                    return
            else:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                rows = [{"filename": fn, **p, "annotator": self.annotator, "timestamp": ts}
                        for p in self.cur_pairs]
                self._append(self.out_path, FIELDNAMES, rows)

        self.n_done_prev += 1
        self._go(+1)

    def _reject(self):
        if self.mode == "answer":
            return
        fn = self.files[self.idx]
        if self.mode == "select":
            self._append(self.plan_path, ["filename", "qid", "question"],
                         [{"filename": fn, "qid": SKIP_QID, "question": ""}])
        else:
            self._append(self.rej_path, ["filename", "reason", "annotator", "timestamp"],
                         [{"filename": fn, "reason": "khong dung duoc",
                           "annotator": self.annotator,
                           "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
        self.n_done_prev += 1
        self._go(+1)

    def _prev(self):
        self._go(-1)

    def _go(self, delta: int):
        new = self.idx + delta
        if new < 0:
            return
        if new >= len(self.files):
            messagebox.showinfo("Hết", "Đã duyệt hết danh sách ảnh cần làm.")
            return
        self.idx = new
        self._load_image()


def main():
    ap = argparse.ArgumentParser(description="Gan nhan VQA bang ngan hang cau hoi (10 loai, v2 Round 3)")
    ap.add_argument("--annotator", required=True, help="Ten nguoi gan nhan (vd: dung)")
    ap.add_argument("--mode", choices=["free", "select", "answer"], default="free",
                    help="free=lam ca; select=chot cau (IAA); answer=tra loi theo ke hoach (IAA)")
    ap.add_argument("--images-dir", default=None, help="Thu muc anh (mac dinh images)")
    ap.add_argument("--plan", default=None,
                    help="File ke hoach IAA (mac dinh qa/iaa_plan.csv). "
                         "Mode answer co the nhan NHIEU file ngan cach dau phay, vd: "
                         "qa/iaa_plan_dung.csv,qa/iaa_plan_ban.csv")
    ap.add_argument("--reverse", action="store_true", help="Duyet tu cuoi ve dau")
    ap.add_argument("--round", type=int, default=None,
                    help="So thu tu vong IAA (1..N). Doc danh sach anh qa/rounds/round_<N>.txt, "
                         "ghi plan/ket qua rieng cho vong do.")
    args = ap.parse_args()

    images_dir = Path(args.images_dir) if args.images_dir else IMAGES_DIR
    if args.plan:
        plan_arg = args.plan
    elif args.round:
        plan_arg = str(QA_DIR / "rounds" / f"plan_round_{args.round}.csv")
    else:
        plan_arg = str(PLAN_PATH)

    root = tk.Tk()
    AnnotateApp(root, args.annotator, images_dir, args.reverse, args.mode, plan_arg, args.round)
    root.mainloop()


if __name__ == "__main__":
    main()

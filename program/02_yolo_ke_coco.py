"""Mengubah dataset format YOLO menjadi satu file COCO JSON.

YOLO : class_id cx cy w h  (dinormalisasi 0-1, titik tengah)
COCO : bbox = [x_kiri, y_atas, lebar, tinggi]  (piksel, pojok kiri atas)

Pemakaian : python 02_yolo_ke_coco.py [folder_dataset] [file_keluaran]
"""
import json
import sys
from pathlib import Path

import cv2

EXT = {".jpg", ".jpeg", ".png"}
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset_yolo")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "annotations_coco.json"
KELAS = [k.strip() for k in (ROOT / "classes.txt").read_text(encoding="utf-8").splitlines()
         if k.strip()]

coco = {"images": [], "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(KELAS)]}  # COCO: id mulai 1
ann_id = 1
gambar = sorted(g for g in (ROOT / "images").rglob("*") if g.suffix.lower() in EXT)

for img_id, img_path in enumerate(gambar, 1):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[!] gambar tidak bisa dibaca: {img_path.name}")
        continue
    h, w = img.shape[:2]
    rel = img_path.relative_to(ROOT / "images")
    coco["images"].append({"id": img_id, "file_name": rel.as_posix(), "width": w, "height": h})
    lbl = ROOT / "labels" / rel.with_suffix(".txt")
    if not lbl.exists():
        continue
    for baris in lbl.read_text().splitlines():
        v = baris.split()
        if len(v) != 5:
            continue
        cid, (cx, cy, bw, bh) = int(v[0]), map(float, v[1:])
        x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
        lebar, tinggi = bw * w, bh * h
        coco["annotations"].append({
            "id": ann_id, "image_id": img_id, "category_id": cid + 1,
            "bbox": [round(x, 2), round(y, 2), round(lebar, 2), round(tinggi, 2)],
            "area": round(lebar * tinggi, 2), "iscrowd": 0})
        ann_id += 1

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(coco, f)
print(f"Gambar : {len(coco['images'])}")
print(f"Kotak  : {len(coco['annotations'])}")
print(f"Kelas  : {KELAS}")
print(f"Tersimpan: {OUT}")

"""Memeriksa anotasi YOLO dan menggambar kotaknya agar bisa dicek dengan mata.

Pemakaian : python 01_cek_anotasi.py [folder_dataset] [--gambar 30] [--min-sisi 10]
Keluaran  : folder_dataset/cek_visual/   gambar berkotak (contoh acak, default 30)
            folder_dataset/cek_visual/laporan_masalah.txt   daftar lengkap masalah
Yang diperiksa: file label hilang, format 5 angka, class_id valid, nilai 0-1,
                dan kotak yang sangat kecil (peringatan, bukan kesalahan fatal).
Simpan keluaran terminal untuk laporan:
  python 01_cek_anotasi.py dataset_yolo | Tee-Object log_01.txt
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import cv2

EXT = {".jpg", ".jpeg", ".png"}
WARNA = [(0, 200, 0), (255, 120, 0), (0, 120, 255), (200, 0, 200), (0, 200, 200), (200, 200, 0)]

p = argparse.ArgumentParser(description="Cek anotasi YOLO")
p.add_argument("folder", nargs="?", default="dataset_yolo")
p.add_argument("--gambar", type=int, default=30, help="jumlah gambar yang digambari (0 = semua)")
p.add_argument("--min-sisi", type=int, default=10, help="sisi kotak minimum (piksel)")
a = p.parse_args()

ROOT = Path(a.folder)
KELAS = [k.strip() for k in (ROOT / "classes.txt").read_text(encoding="utf-8").splitlines()
         if k.strip()]
gambar = sorted(g for g in (ROOT / "images").rglob("*") if g.suffix.lower() in EXT)
if not gambar:
    sys.exit(f"Tidak ada gambar di {ROOT / 'images'}")

contoh = set(gambar) if a.gambar <= 0 else set(
    random.Random(42).sample(gambar, min(a.gambar, len(gambar))))
out_dir = ROOT / "cek_visual"
out_dir.mkdir(exist_ok=True)

hitung = Counter()
masalah = []
tanpa_label = kosong = 0

for img_path in gambar:
    img = cv2.imread(str(img_path))
    if img is None:
        masalah.append(f"{img_path.name}: gambar tidak bisa dibaca")
        continue
    h, w = img.shape[:2]
    lbl = ROOT / "labels" / img_path.relative_to(ROOT / "images").with_suffix(".txt")
    if not lbl.exists():
        tanpa_label += 1
        masalah.append(f"{img_path.name}: tidak ada file label")
        continue
    baris = [b for b in lbl.read_text().splitlines() if b.strip()]
    if not baris:
        kosong += 1                      # boleh: gambar latar tanpa objek
    for i, b in enumerate(baris, 1):
        v = b.split()
        if len(v) != 5:
            masalah.append(f"{lbl.name} baris {i}: harus 5 angka, ada {len(v)}")
            continue
        try:
            cid, (cx, cy, bw, bh) = int(v[0]), map(float, v[1:])
        except ValueError:
            masalah.append(f"{lbl.name} baris {i}: ada nilai yang bukan angka")
            continue
        if not 0 <= cid < len(KELAS):
            masalah.append(f"{lbl.name} baris {i}: class_id {cid} tidak ada di classes.txt")
            continue
        if not all(0 <= t <= 1 for t in (cx, cy, bw, bh)):
            masalah.append(f"{lbl.name} baris {i}: nilai di luar 0-1 (belum dinormalisasi?)")
            continue
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        if bw * w < a.min_sisi or bh * h < a.min_sisi:
            masalah.append(f"{lbl.name} baris {i}: kotak sangat kecil "
                           f"({bw * w:.0f}x{bh * h:.0f} px)")
        hitung[KELAS[cid]] += 1
        if img_path in contoh:
            warna = WARNA[cid % len(WARNA)]
            cv2.rectangle(img, (x1, y1), (x2, y2), warna, 2)
            cv2.putText(img, KELAS[cid], (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna, 2)
    if img_path in contoh:
        cv2.imwrite(str(out_dir / img_path.name), img)

print(f"Gambar          : {len(gambar)}")
print(f"Tanpa file label: {tanpa_label}")
print(f"Label kosong    : {kosong} (gambar latar tanpa objek)")
print("Jumlah kotak per kelas:")
for k in KELAS:
    print(f"  {k:10s}: {hitung[k]}")

(out_dir / "laporan_masalah.txt").write_text("\n".join(masalah), encoding="utf-8")
print(f"\n{len(masalah)} masalah ditemukan:")
for m in masalah[:30]:
    print(f"  [!] {m}")
if len(masalah) > 30:
    print(f"  ... dan {len(masalah) - 30} lainnya (lihat {out_dir / 'laporan_masalah.txt'})")
print(f"\nGambar cek visual: {len(contoh)} file di {out_dir}/ (pilih 3 untuk dikumpulkan)")
print(f"Siap di {ROOT}/ -> lanjutkan dengan: python 05_split_dataset_yolo.py {ROOT}")

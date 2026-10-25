"""Membagi dataset YOLO menjadi train/val/test dan membuat data.yaml.

Pemakaian : python 05_split_dataset_yolo.py [folder_sumber] [folder_tujuan]
            (default: dataset_yolo -> dataset_v1)

- Dataset datar (images/*.jpg)  -> dibagi acak 70/20/10 dengan seed 42 (bisa diulang).
- Dataset yang SUDAH terbagi (images/train, images/val, images/test, mis. dari 03 pada
  export Roboflow) -> pembagian asli dipertahankan, supaya gambar hasil augmentasi
  yang sama tidak bocor ke val/test.
Simpan keluaran terminal untuk laporan:
  python 05_split_dataset_yolo.py dataset_yolo | Tee-Object log_05.txt
"""
import random
import shutil
import sys
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png"}
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset_yolo")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "dataset_v1")
PROPORSI = {"train": 0.7, "val": 0.2, "test": 0.1}
random.seed(42)                      # pembagian bisa diulang dengan hasil sama

kelas = [k.strip() for k in (SRC / "classes.txt").read_text(encoding="utf-8").splitlines()
         if k.strip()]


def label_dari(g):
    return SRC / "labels" / g.relative_to(SRC / "images").with_suffix(".txt")


gambar = sorted(p for p in (SRC / "images").rglob("*") if p.suffix.lower() in EXT)
gambar = [g for g in gambar if label_dari(g).exists()]       # hanya yang berlabel
if not gambar:
    sys.exit(f"Tidak ada gambar berlabel di {SRC}")

sub = {g.relative_to(SRC / "images").parts[0] for g in gambar
       if len(g.relative_to(SRC / "images").parts) > 1}
if sub:
    if not {"train", "val", "test"} <= sub:
        sys.exit(f"Split di {SRC} tidak lengkap (ada: {sorted(sub)}); butuh train, val, test.")
    bagian = {n: [g for g in gambar if g.relative_to(SRC / "images").parts[0] == n]
              for n in ("train", "val", "test")}
    print("Pembagian asli train/val/test ditemukan dan dipertahankan.")
else:
    random.shuffle(gambar)
    n = len(gambar)
    n_train = int(n * PROPORSI["train"])
    n_val = int(n * PROPORSI["val"])
    bagian = {"train": gambar[:n_train],
              "val": gambar[n_train:n_train + n_val],
              "test": gambar[n_train + n_val:]}
    print(f"Pembagian acak {PROPORSI} dari {n} gambar (seed 42).")

for lama in ("images", "labels"):
    if (DST / lama).exists():
        shutil.rmtree(DST / lama)

for nama, daftar in bagian.items():
    (DST / "images" / nama).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / nama).mkdir(parents=True, exist_ok=True)
    kotak = 0
    for g in daftar:
        shutil.copy(g, DST / "images" / nama / g.name)
        shutil.copy(label_dari(g), DST / "labels" / nama / f"{g.stem}.txt")
        kotak += sum(1 for b in label_dari(g).read_text().splitlines() if b.strip())
    print(f"{nama:5s}: {len(daftar)} gambar, {kotak} kotak")

yaml = [f"path: {DST.resolve().as_posix()}", "train: images/train", "val: images/val",
        "test: images/test", "names:"] + [f"  {i}: {k}" for i, k in enumerate(kelas)]
(DST / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
print(f"Kelas: {kelas}")
print(f"Siap di {DST}/ -> python 06_latih_bandingkan.py --nim ... --nama \"...\"")

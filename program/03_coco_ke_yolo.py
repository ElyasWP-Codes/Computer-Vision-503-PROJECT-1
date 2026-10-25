"""Mengubah dataset COCO JSON menjadi format YOLO.

Pemakaian:
  python 03_coco_ke_yolo.py folder_roboflow [folder_keluaran]
      folder berisi train/ valid/ test/ masing-masing dengan _annotations.coco.json
      (export Roboflow format COCO). Pembagian train/val/test ASLI dipertahankan.
  python 03_coco_ke_yolo.py file.json [folder_keluaran] [folder_gambar]
      satu file COCO JSON (mis. hasil 02). Folder gambar default: images/ di sebelah file.

Keluaran: folder_keluaran/images, folder_keluaran/labels, folder_keluaran/classes.txt
COCO: bbox [x_kiri, y_atas, lebar, tinggi] piksel -> YOLO: cx cy w h dinormalisasi 0-1.
"""
import json
import shutil
import sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit(__doc__)
SRC = Path(sys.argv[1])
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dataset_yolo")
SPLIT = {"train": "train", "valid": "val", "val": "val", "test": "test"}
names = []


def daftar_kelas(coco):
    """Roboflow menambah kategori dummy (supercategory 'none'); lewati bila ada."""
    cats = coco["categories"]
    asli = [c for c in cats if c.get("supercategory", "none") != "none"] or cats
    return sorted(asli, key=lambda c: c["id"])


def konversi(json_path, img_dir, split):
    coco = json.load(open(json_path, encoding="utf-8"))
    id_map = {}
    for c in daftar_kelas(coco):
        if c["name"] not in names:
            names.append(c["name"])
        id_map[c["id"]] = names.index(c["name"])
    per_gambar = {}
    for an in coco["annotations"]:
        if an["category_id"] in id_map and not an.get("iscrowd", 0):
            per_gambar.setdefault(an["image_id"], []).append(an)

    sub = Path(split) if split else Path()
    (DST / "images" / sub).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / sub).mkdir(parents=True, exist_ok=True)
    n_img = n_box = hilang = 0
    for im in coco["images"]:
        asal = img_dir / im["file_name"]
        if not asal.exists():
            hilang += 1
            continue
        nama = Path(im["file_name"]).name
        shutil.copy(asal, DST / "images" / sub / nama)
        W, H = im["width"], im["height"]
        baris = []
        for an in per_gambar.get(im["id"], []):
            x, y, w, h = an["bbox"]
            cx = min(max((x + w / 2) / W, 0), 1)
            cy = min(max((y + h / 2) / H, 0), 1)
            baris.append(f"{id_map[an['category_id']]} {cx:.6f} {cy:.6f} "
                         f"{min(w / W, 1):.6f} {min(h / H, 1):.6f}")
            n_box += 1
        (DST / "labels" / sub / (Path(nama).stem + ".txt")).write_text("\n".join(baris))
        n_img += 1
    label = split or "semua"
    print(f"{label:6s}: {n_img} gambar, {n_box} kotak" + (f" ({hilang} gambar hilang)" if hilang else ""))


if SRC.is_dir():
    ditemukan = False
    for nama_folder, split in SPLIT.items():
        j = SRC / nama_folder / "_annotations.coco.json"
        if j.exists():
            konversi(j, SRC / nama_folder, split)
            ditemukan = True
    if not ditemukan:
        sys.exit(f"Tidak ada <split>/_annotations.coco.json di {SRC}")
    print("Pembagian train/val/test asli dipertahankan.")
else:
    if len(sys.argv) > 3:
        gambar_dir = Path(sys.argv[3])
    else:
        gambar_dir = SRC.parent / "images" if (SRC.parent / "images").exists() else SRC.parent
    konversi(SRC, gambar_dir, "")

(DST / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
print(f"Kelas: {names}")
print(f"Siap di {DST}/ -> lanjutkan dengan: python 01_cek_anotasi.py {DST}")

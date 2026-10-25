"""Mengambil subset kecil dari COCO val2017 dan menyimpannya dalam format YOLO.

Pemakaian:
  python 00_ambil_subset_coco.py
  python 00_ambil_subset_coco.py --kelas person,car,bottle --jumlah 200 --latar 20 --keluar dataset_yolo

Unduhan (sekitar 1 GB) disimpan di coco_unduhan/ dan tidak diunduh ulang bila sudah ada.
Keluaran: dataset_yolo/images, dataset_yolo/labels, dataset_yolo/classes.txt
(Lewati program ini bila kamu memakai dataset sendiri, mis. dari Roboflow -> pakai 03.)
"""
import argparse
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = {
    "annotations_trainval2017.zip":
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
}

p = argparse.ArgumentParser(description="Ambil subset COCO val2017 -> format YOLO")
p.add_argument("--kelas", default="person,car,bottle", help="nama kelas COCO, pisah koma")
p.add_argument("--jumlah", type=int, default=200, help="jumlah gambar yang berobjek")
p.add_argument("--latar", type=int, default=20, help="jumlah gambar latar (tanpa objek)")
p.add_argument("--keluar", default="dataset_yolo")
p.add_argument("--unduhan", default="coco_unduhan")
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()


def unduh(nama, folder):
    tujuan = folder / nama
    if tujuan.exists():
        print(f"[ada] {tujuan}")
        return tujuan
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Mengunduh {nama} ...")

    def hook(blok, ukuran, total):
        if total > 0:
            print(f"\r  {min(100, blok * ukuran * 100 // total)}%", end="", flush=True)

    sementara = tujuan.with_suffix(".part")
    urllib.request.urlretrieve(URL[nama], sementara, hook)
    sementara.replace(tujuan)
    print()
    return tujuan


kelas = [k.strip() for k in a.kelas.split(",") if k.strip()]
folder = Path(a.unduhan)
zip_ann = unduh("annotations_trainval2017.zip", folder)
zip_img = unduh("val2017.zip", folder)

with zipfile.ZipFile(zip_ann) as z:
    with z.open("annotations/instances_val2017.json") as f:
        data = json.load(f)

nama2id = {c["name"]: c["id"] for c in data["categories"]}
salah = [k for k in kelas if k not in nama2id]
if salah:
    sys.exit(f"Kelas tidak ada di COCO: {salah}. Contoh yang valid: person, car, bottle, dog, cat")
id2idx = {nama2id[k]: i for i, k in enumerate(kelas)}

per_gambar = {}
for an in data["annotations"]:
    if an["category_id"] in id2idx and not an.get("iscrowd", 0):
        per_gambar.setdefault(an["image_id"], []).append(an)
info = {im["id"]: im for im in data["images"]}

rnd = random.Random(a.seed)
berobjek = sorted(per_gambar)
rnd.shuffle(berobjek)
pilih = berobjek[:a.jumlah]
tanpa = sorted(set(info) - set(per_gambar))
rnd.shuffle(tanpa)
latar = tanpa[:a.latar]

dst = Path(a.keluar)
(dst / "images").mkdir(parents=True, exist_ok=True)
(dst / "labels").mkdir(parents=True, exist_ok=True)
hitung = {k: 0 for k in kelas}

with zipfile.ZipFile(zip_img) as z:
    for iid in pilih + latar:
        im = info[iid]
        nama = im["file_name"]
        W, H = im["width"], im["height"]
        (dst / "images" / nama).write_bytes(z.read(f"val2017/{nama}"))
        baris = []
        for an in per_gambar.get(iid, []):
            x, y, w, h = an["bbox"]
            if w <= 1 or h <= 1:
                continue
            cx = min(max((x + w / 2) / W, 0), 1)
            cy = min(max((y + h / 2) / H, 0), 1)
            baris.append(f"{id2idx[an['category_id']]} {cx:.6f} {cy:.6f} "
                         f"{min(w / W, 1):.6f} {min(h / H, 1):.6f}")
            hitung[kelas[id2idx[an['category_id']]]] += 1
        (dst / "labels" / (Path(nama).stem + ".txt")).write_text("\n".join(baris))

(dst / "classes.txt").write_text("\n".join(kelas) + "\n", encoding="utf-8")

print(f"Gambar berobjek : {len(pilih)}")
print(f"Gambar latar    : {len(latar)}")
print("Jumlah kotak    : " + ", ".join(f"{k} {n}" for k, n in hitung.items()))
print(f"Siap di {dst}/ -> lanjutkan dengan: python 01_cek_anotasi.py {dst}")

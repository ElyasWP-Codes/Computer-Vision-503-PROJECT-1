"""Praktik terpadu minggu 4-7, tahap 3-5: melatih, tuning, memilih, dan menguji model YOLO.

Eksperimen yang dijalankan (pengaturan lain SAMA untuk semua):
  E1 pretrained : model pretrained COCO (transfer learning)   -> baseline
  E2 scratch    : arsitektur sama, bobot acak (tanpa transfer) -> pembanding
  E3 tuning     : E1 + SATU perubahan (default imgsz=640)      -> eksperimen tuning
Model terbaik dipilih dengan mAP50-95 pada data VAL, lalu diuji SEKALI pada data TEST.

Pemakaian:
  python 06_latih_bandingkan.py --nim 4222401008 --nama "Nama Kamu"
  python 06_latih_bandingkan.py --nim ... --nama "..." --epochs 30 --ubah epochs=60
  python 06_latih_bandingkan.py --nim ... --nama "..." --model yolov8n

Membutuhkan: pip install ultralytics pandas, dan data.yaml format YOLO
(default: dataset_v1/data.yaml, hasil 05_split_dataset_yolo.py). Disarankan memakai GPU.
"""
import argparse
import ast
import csv
import getpass
import hashlib
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

p = argparse.ArgumentParser(description="Latih dan bandingkan E1/E2/E3 (YOLO)")
p.add_argument("--nim", required=True)
p.add_argument("--nama", required=True)
p.add_argument("--data", default="dataset_v1/data.yaml")
p.add_argument("--model", default="yolov8n",
               help="nama model tanpa ekstensi, mis. yolov8n (atau yolo26n bila tersedia)")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--ubah", default=None,
               help="SATU perubahan untuk E3, format kunci=nilai, mis. epochs=60 atau imgsz=800")
args = p.parse_args()

PROJECT = Path("runs_praktik").resolve()
COLS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def parse_ubah(s):
    if "=" not in s:
        sys.exit("--ubah harus berformat kunci=nilai, contoh: epochs=60")
    k, v = s.split("=", 1)
    try:
        v = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        pass  # biarkan sebagai string
    return k.strip(), v


def latih(nama, bobot, pretrained, extra):
    """Latih satu eksperimen, salin hasil ke folder <nama>/, kembalikan ringkasan."""
    print(f"\n{'=' * 60}\n{nama}: bobot={bobot}, pretrained={pretrained}, ubah={extra}\n{'=' * 60}")
    model = YOLO(bobot)
    kw = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
              seed=args.seed, workers=args.workers, project=str(PROJECT), name=nama,
              exist_ok=True, pretrained=pretrained, plots=True)
    kw.update(extra)  # satu perubahan untuk E3 menimpa nilai default di atas

    t0 = time.time()
    model.train(**kw)
    durasi = time.time() - t0

    save_dir = Path(model.trainer.save_dir)
    out = Path(nama)
    out.mkdir(exist_ok=True)
    for f in ("results.csv", "results.png", "confusion_matrix.png"):
        if (save_dir / f).exists():
            shutil.copy(save_dir / f, out / f)

    df = pd.read_csv(save_dir / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    best = df.loc[df[COLS["mAP50-95"]].idxmax()]
    hasil = {"eksperimen": nama, "epoch_terbaik": int(best["epoch"]),
             "durasi_menit": round(durasi / 60, 1)}
    for k, c in COLS.items():
        hasil[k] = round(float(best[c]), 4)
    hasil["best_pt"] = str(save_dir / "weights" / "best.pt")
    return hasil


def main():
    if not Path(args.data).exists():
        sys.exit(f"data.yaml tidak ditemukan: {args.data}")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "TIDAK ADA (CPU)"
    print("GPU:", gpu)
    if gpu.startswith("TIDAK"):
        print("[!] GPU tidak terdeteksi, training akan sangat lambat.")

    ubah = parse_ubah(args.ubah) if args.ubah else ("epochs", args.epochs * 2)
    extra_e3 = {ubah[0]: ubah[1]}

    hasil = [
        latih("E1", f"{args.model}.pt", True, {}),
        latih("E2", f"{args.model}.yaml", False, {}),
        latih("E3", f"{args.model}.pt", True, extra_e3),
    ]

    # Pilih model terbaik berdasarkan mAP50-95 di VAL
    terbaik = max(hasil, key=lambda r: r["mAP50-95"])
    print(f"\nModel terbaik (val, mAP50-95): {terbaik['eksperimen']} = {terbaik['mAP50-95']}")

    # Uji SEKALI pada data TEST
    m = YOLO(terbaik["best_pt"]).val(
        data=args.data, split="test", imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, project=str(PROJECT), name="uji_test", exist_ok=True, plots=True)
    uji = {"eksperimen": f"TEST ({terbaik['eksperimen']})", "epoch_terbaik": "",
           "durasi_menit": "", "precision": round(float(m.box.mp), 4),
           "recall": round(float(m.box.mr), 4), "mAP50": round(float(m.box.map50), 4),
           "mAP50-95": round(float(m.box.map), 4), "best_pt": terbaik["best_pt"]}
    cm = Path(m.save_dir) / "confusion_matrix.png"
    if cm.exists():
        Path("uji_test").mkdir(exist_ok=True)
        shutil.copy(cm, "uji_test/confusion_matrix.png")

    # Tulis hasil_praktik_<NIM>.csv dan .txt
    baris = hasil + [uji]
    kolom = ["eksperimen", "epoch_terbaik", "precision", "recall", "mAP50",
             "mAP50-95", "durasi_menit"]
    with open(f"hasil_praktik_{args.nim}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=kolom, extrasaction="ignore")
        w.writeheader()
        w.writerows(baris)

    sidik = hashlib.md5(Path(args.data).read_bytes()).hexdigest()[:10]
    with open(f"hasil_praktik_{args.nim}.txt", "w", encoding="utf-8") as fh:
        fh.write(f"NIM            : {args.nim}\nNama           : {args.nama}\n")
        fh.write(f"Waktu          : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        fh.write(f"Pengguna/Mesin : {getpass.getuser()} / {platform.node()}\n")
        fh.write(f"GPU            : {gpu}\n")
        fh.write(f"Model          : {args.model}\nEpochs         : {args.epochs}\n")
        fh.write(f"imgsz / batch  : {args.imgsz} / {args.batch}\nSeed           : {args.seed}\n")
        fh.write(f"Perubahan E3   : {ubah[0]}={ubah[1]}\nHash data.yaml : {sidik}\n\n")
        fh.write(pd.DataFrame(baris)[kolom].to_string(index=False))
        fh.write(f"\n\nModel terbaik (val): {terbaik['eksperimen']}\n")

    print("\n" + pd.DataFrame(baris)[kolom].to_string(index=False))
    print(f"\nSelesai. File: hasil_praktik_{args.nim}.csv/.txt dan folder E1, E2, E3, uji_test")


if __name__ == "__main__":
    main()

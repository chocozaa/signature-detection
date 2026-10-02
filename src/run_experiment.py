"""
Menjalankan seluruh eksperimen:
  * 9 citra DENGAN tanda tangan  (data/with_signature)
  * 9 citra TANPA tanda tangan   (data/no_signature, dibuat oleh make_negatives.py)
  * 9 potongan kertas polos      (ROI "blank" dari citra asli, tanpa tinta sama sekali)
  * 3 metode threshold: global, otsu, adaptive

Output di folder outputs/:
  results.csv                 -> fitur + keputusan per citra per metode
  summary.md                  -> tabel akurasi per metode
  crops/                      -> ROI hasil crop
  pipeline/<nama>.png         -> crop -> gray -> threshold -> morfologi
  compare_methods.png         -> perbandingan 3 metode untuk semua citra
  threshold_sweep.png         -> efek threshold terlalu rendah / tinggi
  threshold_extremes.png      -> contoh visual T rendah / sedang / tinggi
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detector as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


class Sample:
    def __init__(self, path: Path, label: str, roi: str):
        self.path, self.label, self.roi = path, label, roi
        self.id = path.stem + ("_BLANK" if roi == "blank" else "")
        self.name = (path.stem.replace("_1_NOSIG", " (no sig)").replace("_1", "")
                     + (" (blank)" if roi == "blank" else ""))


def load_dataset():
    items = []
    for label, folder in (("PRESENT", "with_signature"), ("ABSENT", "no_signature")):
        for p in sorted((ROOT / "data" / folder).glob("*.jpg")):
            items.append(Sample(p, label, "dekan"))
    for p in sorted((ROOT / "data" / "with_signature").glob("*.jpg")):
        items.append(Sample(p, "ABSENT", "blank"))
    return items


def save_pipeline_figure(name, res, otsu_t, path):
    titles = ["ROI (crop)", "Grayscale", "Threshold mentah", "Opening+Closing", "Final (filter komponen)"]
    imgs = [cv2.cvtColor(res["crop"], cv2.COLOR_BGR2RGB), res["gray"], res["binary"], res["morph"], res["clean"]]
    fig, ax = plt.subplots(1, 5, figsize=(14, 5.5))
    for a, im, t in zip(ax, imgs, titles):
        a.imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=255)
        a.set_title(t, fontsize=10)
        a.axis("off")
    f = res["features"]
    fig.suptitle(f"{name}  |  fg={f.fg_pixels}  ratio={f.ink_ratio:.3f}  "
                 f"largest={f.largest_component}  ->  {res['decision']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Eksperimen deteksi tanda tangan")
    ap.add_argument("--method", default="adaptive", choices=D.METHODS,
                    help="metode yang dipakai untuk gambar pipeline per-citra")
    args = ap.parse_args(argv)

    for sub in ("crops", "pipeline"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    data = load_dataset()
    if not data:
        sys.exit("Dataset kosong. Jalankan dulu: python src/make_negatives.py")

    rows = []
    grid = []  # untuk compare_methods.png
    for s in data:
        path, label = s.path, s.label
        img = cv2.imread(str(path))
        crop = D.crop_roi(img, s.roi)
        cv2.imwrite(str(OUT / "crops" / f"{s.id}.png"), crop)
        _, otsu_t = D.threshold_otsu(D.denoise(D.to_gray(crop)))
        row_imgs = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)]
        for m in D.METHODS:
            r = D.analyze(crop, method=m, roi_already_cropped=True)
            f = r["features"]
            pred = "PRESENT" if r["decision"] == "SIGNATURE PRESENT" else "ABSENT"
            rows.append({
                "image": s.id, "ground_truth": label, "method": m,
                "otsu_T": round(otsu_t, 1) if m == "otsu" else "",
                "raw_fg_pixels": r["raw_fg_pixels"], **f.as_dict(),
                "decision": r["decision"], "correct": pred == label,
            })
            row_imgs.append(r["clean"])
            if m == args.method:
                save_pipeline_figure(s.name, r, otsu_t, OUT / "pipeline" / f"{s.id}.png")
        grid.append((s.name, row_imgs))

    # ---------- CSV ----------
    with open(OUT / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["ink_ratio"] = round(r["ink_ratio"], 5)
            r["bbox_coverage"] = round(r["bbox_coverage"], 4)
            w.writerow(r)

    # ---------- Ringkasan ----------
    lines = ["| Metode | Benar | Akurasi | TP | TN | FP | FN |", "|---|---|---|---|---|---|---|"]
    print("\n=== Ringkasan per metode ===")
    for m in D.METHODS:
        rs = [r for r in rows if r["method"] == m]
        tp = sum(r["ground_truth"] == "PRESENT" and r["correct"] for r in rs)
        tn = sum(r["ground_truth"] == "ABSENT" and r["correct"] for r in rs)
        fp = sum(r["ground_truth"] == "ABSENT" and not r["correct"] for r in rs)
        fn = sum(r["ground_truth"] == "PRESENT" and not r["correct"] for r in rs)
        acc = (tp + tn) / len(rs)
        lines.append(f"| {m} | {tp+tn}/{len(rs)} | {acc:.0%} | {tp} | {tn} | {fp} | {fn} |")
        print(f"{m:9s} akurasi {acc:6.1%}  (TP={tp} TN={tn} FP={fp} FN={fn})")
    print("\n=== Detail (metode %s) ===" % args.method)
    for r in rows:
        if r["method"] == args.method:
            mark = "OK " if r["correct"] else "SALAH"
            print(f"[{mark}] {r['image']:44s} GT={r['ground_truth']:7s} "
                  f"fg={r['fg_pixels']:6d} ratio={r['ink_ratio']:.4f} "
                  f"largest={r['largest_component']:6d} -> {r['decision']}")

    detail = ["", "| Citra | GT | " + " | ".join(f"{m} fg / keputusan" for m in D.METHODS) + " |",
              "|---|---|" + "---|" * len(D.METHODS)]
    for s in data:
        cells = []
        for m in D.METHODS:
            r = next(x for x in rows if x["image"] == s.id and x["method"] == m)
            ok = "✅" if r["correct"] else "❌"
            cells.append(f"{r['fg_pixels']} / {r['decision'].split()[1]} {ok}")
        detail.append(f"| {s.name} | {s.label} | " + " | ".join(cells) + " |")
    (OUT / "summary.md").write_text("\n".join(lines + detail) + "\n")

    # ---------- Perbandingan metode ----------
    n = len(grid)
    fig, ax = plt.subplots(n, 4, figsize=(14, 1.75 * n))
    heads = ["ROI", "Global (T=%d)" % D.GLOBAL_T, "Otsu", "Adaptive"]
    for i, (name, ims) in enumerate(grid):
        for j, im in enumerate(ims):
            a = ax[i, j]
            a.imshow(np.rot90(im, -1), cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=255)
            a.set_xticks([]); a.set_yticks([])
            if i == 0:
                a.set_title(heads[j], fontsize=10)
            if j == 0:
                a.set_ylabel(name, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "compare_methods.png", dpi=80)
    plt.close(fig)

    # ---------- Sweep threshold global ----------
    ts = list(range(20, 251, 10))
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in data:
        label = s.label
        g = D.denoise(D.to_gray(D.crop_roi(cv2.imread(str(s.path)), s.roi)))
        ratios = []
        for t in ts:
            b = D.remove_small_components(D.apply_morphology(D.threshold_global(g, t)))
            ratios.append(np.count_nonzero(b) / b.size)
        ax.plot(ts, ratios, color={"PRESENT": "tab:blue"}.get(label, "tab:red" if s.roi == "dekan" else "tab:orange"),
                alpha=0.7, lw=1.4)
    ax.axhline(D.MIN_INK_RATIO, color="k", ls="--", lw=1, label=f"MIN_INK_RATIO={D.MIN_INK_RATIO}")
    ax.axvline(D.GLOBAL_T, color="gray", ls=":", lw=1, label=f"T global={D.GLOBAL_T}")
    ax.plot([], [], color="tab:blue", label="dengan tanda tangan")
    ax.plot([], [], color="tab:red", label="tanpa tanda tangan")
    ax.plot([], [], color="tab:orange", label="kertas polos")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("Nilai threshold global T")
    ax.set_ylabel("Rasio piksel foreground (setelah morfologi)")
    ax.set_title("Efek nilai threshold global terhadap foreground")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_sweep.png", dpi=100)
    plt.close(fig)

    # ---------- Contoh T terlalu rendah / pas / terlalu tinggi ----------
    samples = [s for s in data if s.label == "PRESENT" and s.path.stem.startswith(("01_", "02_", "06_"))]
    fig, ax = plt.subplots(len(samples), 4, figsize=(14, 2.1 * len(samples)))
    for i, s in enumerate(samples):
        g = D.denoise(D.to_gray(D.crop_roi(cv2.imread(str(s.path)))))
        ax[i, 0].imshow(np.rot90(g, -1), cmap="gray", vmin=0, vmax=255); ax[i, 0].set_ylabel(s.name, fontsize=8)
        for j, t in enumerate((50, 127, 220), start=1):
            ax[i, j].imshow(np.rot90(D.threshold_global(g, t), -1), cmap="gray", vmin=0, vmax=255)
            if i == 0:
                ax[i, j].set_title(["", "T=50 (terlalu rendah)", "T=127", "T=220 (terlalu tinggi)"][j], fontsize=9)
        ax[0, 0].set_title("Grayscale", fontsize=9)
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "threshold_extremes.png", dpi=90)
    plt.close(fig)

    print(f"\nSemua output tersimpan di {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()

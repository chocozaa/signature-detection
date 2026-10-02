"""
Deteksi tanda tangan untuk satu atau beberapa citra.

Contoh:
    python src/detect.py data/with_signature/01_HighQuality_Enhanced_1.jpg
    python src/detect.py data/no_signature/*.jpg --method otsu
    python src/detect.py scan_baru.jpg --roi rektor --save hasil.png
"""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detector as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="SIGNATURE PRESENT / ABSENT detector")
    ap.add_argument("images", nargs="+", help="path citra scan dokumen")
    ap.add_argument("--method", default="adaptive", choices=D.METHODS)
    ap.add_argument("--roi", default="dekan", choices=list(D.ROIS))
    ap.add_argument("--cropped", action="store_true",
                    help="input sudah berupa potongan area tanda tangan")
    ap.add_argument("--save", help="simpan visualisasi ROI|biner (hanya untuk 1 citra)")
    a = ap.parse_args()

    for p in a.images:
        img = cv2.imread(p)
        if img is None:
            print(f"{p}: gagal dibaca")
            continue
        r = D.analyze(img, method=a.method, roi=a.roi, roi_already_cropped=a.cropped)
        f = r["features"]
        print(f"{p}\n  metode={a.method}  fg_pixels={f.fg_pixels}  ink_ratio={f.ink_ratio:.4f}  "
              f"komponen={f.n_components}  komponen_terbesar={f.largest_component}\n"
              f"  ==> {r['decision']}")
        if a.save and len(a.images) == 1:
            vis = np.hstack([r["crop"], cv2.cvtColor(r["clean"], cv2.COLOR_GRAY2BGR)])
            cv2.putText(vis, r["decision"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)
            cv2.imwrite(a.save, vis)
            print("  visualisasi disimpan ke", a.save)


if __name__ == "__main__":
    main()

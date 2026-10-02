"""
Membuat citra uji TANPA tanda tangan (kelas negatif).

Tidak ada scan asli tanpa tanda tangan, jadi kelas negatif disintesis dari
setiap scan: goresan tanda tangan di dalam ROI dihapus dengan inpainting
(warna kertas di sekitarnya mengisi bekas goresan), lalu tekstur noise
kertas ditambahkan kembali agar karakter degradasi tiap citra tetap sama.
Teks cetak "Dekan" sengaja dipertahankan, karena pada formulir kosong
teks itu memang tetap ada.

Hasil: data/no_signature/<nama>_NOSIG.jpg
"""
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import ROIS, REFERENCE_SIZE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "with_signature"
DST = ROOT / "data" / "no_signature"

# Kotak teks cetak "Dekan" relatif terhadap ROI dekan (dipertahankan)
KEEP_BOX = (15, 300, 80, 480)  # x1, y1, x2, y2


def erase_signature(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    sx, sy = w / REFERENCE_SIZE[0], h / REFERENCE_SIZE[1]
    x1, y1, x2, y2 = ROIS["dekan"]
    x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
    roi = img[y1:y2, x1:x2].copy()

    gray = cv2.medianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 3)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 51, 8)
    ink = cv2.dilate(ink, np.ones((9, 9), np.uint8))
    kx1, ky1, kx2, ky2 = KEEP_BOX
    ink[int(ky1 * sy):int(ky2 * sy), int(kx1 * sx):int(kx2 * sx)] = 0

    filled = cv2.inpaint(roi, ink, 7, cv2.INPAINT_TELEA)

    # kembalikan tekstur noise kertas pada area yang di-inpaint
    paper = roi[ink == 0].astype(np.float32)
    std = float(np.clip(paper.std(axis=0).mean() * 0.35, 1.0, 12.0))
    noise = rng.normal(0, std, filled.shape).astype(np.float32)
    mask3 = (ink > 0)[..., None]
    filled = np.where(mask3, np.clip(filled + noise, 0, 255), filled).astype(np.uint8)

    out = img.copy()
    out[y1:y2, x1:x2] = filled
    return out


def main():
    DST.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for p in sorted(SRC.glob("*.jpg")):
        img = cv2.imread(str(p))
        out = erase_signature(img, rng)
        name = DST / f"{p.stem}_NOSIG.jpg"
        cv2.imwrite(str(name), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("dibuat:", name.relative_to(ROOT))


if __name__ == "__main__":
    main()

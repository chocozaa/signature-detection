"""
Inti pipeline deteksi keberadaan tanda tangan.

Tahapan:
    1. crop_roi            -> potong area tanda tangan pejabat penandatangan (Dekan)
    2. to_gray             -> konversi ke grayscale
    3. threshold_*         -> segmentasi tinta vs kertas (global, Otsu, adaptive)
    4. apply_morphology    -> opening (buang noise) + closing (sambung goresan)
    5. extract_features    -> jumlah piksel foreground, rasio, komponen terhubung
    6. decide              -> aturan sederhana SIGNATURE PRESENT / ABSENT
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
# ROI (x1, y1, x2, y2) dalam piksel untuk scan berukuran 2481 x 3506 (A4, 300 dpi).
# Dokumen discan dalam posisi diputar 90 derajat, jadi tanda tangan Dekan
# berada di sisi kanan-atas citra.
REFERENCE_SIZE = (2481, 3506)  # (lebar, tinggi)
ROIS = {
    "dekan": (1650, 440, 2030, 1280),
    "rektor": (1700, 1350, 2000, 2000),
    # area kertas kosong (margin kiri-atas), ukuran sama dgn ROI dekan;
    # dipakai sebagai uji negatif tambahan "kertas polos"
    "blank": (60, 100, 440, 940),
}

GLOBAL_T = 127            # ambang global tetap
ADAPTIVE_BLOCK = 51       # ukuran jendela adaptive threshold (ganjil)
ADAPTIVE_C = 15           # konstanta pengurang adaptive threshold
OPEN_KERNEL = 3           # kernel opening
CLOSE_KERNEL = 5          # kernel closing
MIN_COMPONENT_AREA = 80   # komponen < ini dianggap noise

# Aturan keputusan (dikalibrasi dari data, lihat README)
MIN_INK_RATIO = 0.015     # >= 1.5 % piksel ROI harus tinta
MIN_LARGEST_COMPONENT = 3000  # goresan tanda tangan terpanjang (piksel)


# ---------------------------------------------------------------------------
# 1-2. Crop & grayscale
# ---------------------------------------------------------------------------
def crop_roi(img: np.ndarray, roi: str = "dekan") -> np.ndarray:
    """Potong ROI, diskalakan otomatis bila resolusi citra berbeda dari referensi."""
    h, w = img.shape[:2]
    sx, sy = w / REFERENCE_SIZE[0], h / REFERENCE_SIZE[1]
    x1, y1, x2, y2 = ROIS[roi]
    return img[int(y1 * sy):int(y2 * sy), int(x1 * sx):int(x2 * sx)].copy()


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Median blur ringan: meredam noise salt-and-pepper / grain scan."""
    return cv2.medianBlur(gray, 3)


# ---------------------------------------------------------------------------
# 3. Thresholding  (output: biner, tinta = 255, kertas = 0)
# ---------------------------------------------------------------------------
def threshold_global(gray: np.ndarray, t: int = GLOBAL_T) -> np.ndarray:
    _, b = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
    return b


def threshold_otsu(gray: np.ndarray) -> tuple[np.ndarray, float]:
    t, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b, float(t)


def threshold_adaptive(gray: np.ndarray, block: int = ADAPTIVE_BLOCK,
                       c: int = ADAPTIVE_C) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, c)


METHODS = ("global", "otsu", "adaptive")


def threshold(gray: np.ndarray, method: str) -> np.ndarray:
    if method == "global":
        return threshold_global(gray)
    if method == "otsu":
        return threshold_otsu(gray)[0]
    if method == "adaptive":
        return threshold_adaptive(gray)
    raise ValueError(method)


# ---------------------------------------------------------------------------
# 4. Morfologi
# ---------------------------------------------------------------------------
def apply_morphology(binary: np.ndarray) -> np.ndarray:
    """Opening -> buang bintik noise kecil; Closing -> sambung goresan yang putus."""
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL, OPEN_KERNEL))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
    return closed


def remove_small_components(binary: np.ndarray, min_area: int = MIN_COMPONENT_AREA) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. Fitur
# ---------------------------------------------------------------------------
@dataclass
class Features:
    fg_pixels: int          # jumlah piksel foreground (tinta)
    ink_ratio: float        # fg_pixels / total piksel ROI
    n_components: int       # jumlah komponen terhubung (setelah filter)
    largest_component: int  # luas komponen terbesar
    bbox_coverage: float    # luas bounding-box seluruh tinta / luas ROI

    def as_dict(self):
        return asdict(self)


def extract_features(binary: np.ndarray) -> Features:
    total = binary.size
    fg = int(np.count_nonzero(binary))
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
    if fg:
        ys, xs = np.nonzero(binary)
        bbox = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) / total
    else:
        bbox = 0.0
    return Features(fg, fg / total, max(n - 1, 0), int(areas.max()), float(bbox))


# ---------------------------------------------------------------------------
# 6. Aturan keputusan
# ---------------------------------------------------------------------------
def decide(f: Features) -> str:
    """
    SIGNATURE PRESENT jika:
      - cukup banyak tinta di ROI (ink_ratio >= MIN_INK_RATIO), DAN
      - ada setidaknya satu goresan panjang yang menyambung
        (largest_component >= MIN_LARGEST_COMPONENT).
    Syarat kedua mencegah noise/teks cetak kecil (mis. kata "Dekan")
    yang tersebar dianggap sebagai tanda tangan.
    """
    ok = f.ink_ratio >= MIN_INK_RATIO and f.largest_component >= MIN_LARGEST_COMPONENT
    return "SIGNATURE PRESENT" if ok else "SIGNATURE ABSENT"


def analyze(img: np.ndarray, method: str = "adaptive", roi: str = "dekan",
            roi_already_cropped: bool = False) -> dict:
    """Pipeline lengkap untuk satu citra; mengembalikan semua tahap + hasil."""
    crop = img if roi_already_cropped else crop_roi(img, roi)
    gray = denoise(to_gray(crop))
    raw = threshold(gray, method)
    morph = apply_morphology(raw)
    clean = remove_small_components(morph)
    feats = extract_features(clean)
    return {
        "crop": crop, "gray": gray, "binary": raw, "morph": morph, "clean": clean,
        "features": feats, "raw_fg_pixels": int(np.count_nonzero(raw)),
        "decision": decide(feats),
    }

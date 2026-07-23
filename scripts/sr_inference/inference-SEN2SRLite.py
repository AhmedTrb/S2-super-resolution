import os
import gc
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
import torch
import torch.nn.functional as F
from tqdm import tqdm
import mlstac
import sen2sr

# ============================================================
# CONFIG
# ============================================================

INPUT_TIF  = r"C:/Users/ahtrabelsi/Desktop/stage/S2 data/Sentinel2_T31UEQ_TSG.tif"
OUTPUT_TIF = r"C:/Users/ahtrabelsi/Desktop/stage/sujet/SR_OUTPUTS/timestamp_01_SR.tif"

MODEL_URL = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json"
MODEL_DIR = "model/SEN2SRLite"

UPSCALE      = 4        # SEN2SRLite x4
WINDOW_SIZE  = 1024     # input tile size (must be multiple of MODEL_MULTIPLE)
OVERLAP      = 32       # internal overlap in predict_large

# The model's FFT low-pass mask is sized for chunks that are multiples of
# MODEL_MULTIPLE.  Edge tiles that are smaller are NOT passed to the model —
# instead the original pixels are upsampled with bicubic interpolation so
# the output is spatially consistent (no black or corrupt borders).
MODEL_MULTIPLE = 256    # 128 (mask half-size) × 2

BANDS_PER_IMAGE = 10
BAND_NAMES = ["B02", "B03", "B04", "B05", "B06",
              "B07", "B08", "B8A", "B11", "B12"]

TIMESTAMP_IDX = 0       # 0-based → first timestamp

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# LOAD MODEL
# ============================================================

def load_sr_model(model_url, output_dir, device):
    mlstac.download(file=model_url, output_dir=output_dir)
    model = mlstac.load(output_dir).compiled_model(device=device)
    model.eval()
    return model

print("Loading model …")
SEN2SRLite = load_sr_model(MODEL_URL, MODEL_DIR, device)
print("Model loaded.")

# ============================================================
# HELPERS
# ============================================================

def is_model_compatible(h: int, w: int) -> bool:
    """True only when both spatial dims are multiples of MODEL_MULTIPLE."""
    return (h % MODEL_MULTIPLE == 0) and (w % MODEL_MULTIPLE == 0)


def bicubic_upsample(cube: np.ndarray, scale: int) -> np.ndarray:
    """
    Upsample a (C, H, W) float32 array by `scale` using bicubic interpolation.
    Used as the fallback for border tiles that the SR model cannot handle.
    """
    t = torch.from_numpy(cube).unsqueeze(0)          # (1, C, H, W)
    up = F.interpolate(t, scale_factor=scale, mode="bicubic", align_corners=False)
    return up.squeeze(0).numpy()                      # (C, H*scale, W*scale)

# ============================================================
# OPEN INPUT & PROCESS
# ============================================================

with rasterio.open(INPUT_TIF) as src:

    width  = src.width
    height = src.height
    print(f"Input size: {width} × {height}")

    # ---- band indices for the requested timestamp (1-based) ----
    start_band   = TIMESTAMP_IDX * BANDS_PER_IMAGE + 1
    band_indexes = list(range(start_band, start_band + BANDS_PER_IMAGE))
    print("Reading bands:", band_indexes)

    # ---- output dimensions ----
    out_width  = width  * UPSCALE
    out_height = height * UPSCALE

    t = src.transform
    new_transform = Affine(
        t.a / UPSCALE, t.b, t.c,
        t.d, t.e / UPSCALE, t.f
    )

    profile = src.profile.copy()
    profile.update(
        driver     = "GTiff",
        width      = out_width,
        height     = out_height,
        count      = BANDS_PER_IMAGE,
        dtype      = "float32",
        transform  = new_transform,
        tiled      = True,
        blockxsize = 512,
        blockysize = 512,
        compress   = "zstd",
        BIGTIFF    = "YES",
    )
    profile.pop("interleave", None)

    os.makedirs(os.path.dirname(OUTPUT_TIF), exist_ok=True)

    sr_count      = 0   # tiles processed by the SR model
    bicubic_count = 0   # border tiles upsampled with bicubic fallback

    with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:

        x_positions = list(range(0, width,  WINDOW_SIZE))
        y_positions = list(range(0, height, WINDOW_SIZE))
        pbar = tqdm(total=len(x_positions) * len(y_positions))

        for y in y_positions:
            for x in x_positions:

                # ---- actual tile size (clips at raster edge) ----
                w_in = min(WINDOW_SIZE, width  - x)
                h_in = min(WINDOW_SIZE, height - y)

                in_window = Window(
                    col_off=x, row_off=y,
                    width=w_in, height=h_in
                )

                # ---- read & normalise ----
                cube = src.read(band_indexes, window=in_window).astype(np.float32)
                cube /= 10000.0
                cube = np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0)
                cube = np.clip(cube, 0.0, 1.0)

                # ---- route: SR model or bicubic fallback ----
                if is_model_compatible(h_in, w_in):
                    # Full tile — safe to run through the SR model
                    X = torch.from_numpy(cube).to(device)
                    with torch.no_grad():
                        sr_patch = sen2sr.predict_large(
                            model=SEN2SRLite, X=X, overlap=OVERLAP
                        )
                    if isinstance(sr_patch, torch.Tensor):
                        sr_patch = sr_patch.cpu().numpy()
                    del X
                    sr_count += 1
                else:
                    # Border tile — dimensions not a multiple of MODEL_MULTIPLE.
                    # Upsample with bicubic so the border looks consistent with
                    # the rest of the image (no black strip, no corrupt pixels).
                    sr_patch = bicubic_upsample(cube, UPSCALE)
                    bicubic_count += 1

                sr_patch = np.clip(sr_patch, 0.0, 1.0)
                sr_patch = np.nan_to_num(sr_patch.astype(np.float32))

                # ---- clamp output window to raster edge ----
                ox = x * UPSCALE
                oy = y * UPSCALE
                write_w = min(sr_patch.shape[2], out_width  - ox)
                write_h = min(sr_patch.shape[1], out_height - oy)
                sr_patch = sr_patch[:, :write_h, :write_w]

                out_window = Window(
                    col_off=ox, row_off=oy,
                    width=write_w, height=write_h
                )
                dst.write(sr_patch, window=out_window)

                # ---- free memory ----
                del cube, sr_patch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                pbar.update(1)

        pbar.close()

        # ---- band descriptions ----
        for i, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(i, name)

print(f"\nDone.")
print(f"  SR model tiles : {sr_count}")
print(f"  Bicubic fallback tiles : {bicubic_count}")
print(f"  Output : {OUTPUT_TIF}")
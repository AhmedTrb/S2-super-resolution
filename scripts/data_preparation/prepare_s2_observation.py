import sys
import time
from pathlib import Path
from typing import Union

import geopandas as gpd
import mlstac
import rasterio
import torch
from shapely.geometry import box

repo_root = Path.cwd().resolve()
if not (repo_root / "src").exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root / "src"))

from s2_pipeline import ObservationDrivenS2Pipeline, SuperResolutionProcessor, TemporalDateMatcher


def log_stage(message: str) -> None:
    print(message)


def format_elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def load_sr_model(model_url: str, model_dir: Union[str, Path], device: torch.device) -> torch.nn.Module:
    model_dir = Path(model_dir)
    log_stage(f"[MODEL] Loading SEN2SRLite from {model_dir} on {device}...")
    load_start = time.perf_counter()
    mlstac.download(file=model_url, output_dir=str(model_dir))
    model = mlstac.load(str(model_dir)).compiled_model(device=str(device))
    model.eval()
    log_stage(f"[MODEL] SEN2SRLite ready in {format_elapsed(time.perf_counter() - load_start)}")
    return model


dataset_configs = {
    2020: {
        "parcels_path": r"C:\Users\ahtrabelsi\Desktop\stage\s2-super-resolution\data\shapefiles\2020_SEPIM.shp",
        "stacks": [
            {
                "stack_path": Path(r"C:/Users/ahtrabelsi/Desktop/stage/S2 data/Sentinel2_T31UEQ_TSG.tif"),
                "calendar_path": repo_root / "data" / "Sentinel2_T31UEQ_interpolation_dates.txt",
            },
            {
                "stack_path": Path(r"C:\Users\ahtrabelsi\Desktop\stage\S2 data\Sentinel2_T31UDP_TSG.tif"),
                "calendar_path": repo_root / "data" / "Sentinel2_T31UDP_interpolation_dates.txt",
            },
        ],
    }
}

out_dir = repo_root / "yellowness_dataset"
out_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEN2SRLite = load_sr_model(
    model_url="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json",
    model_dir=repo_root / "model" / "SEN2SRLite",
    device=device,
)

sr_processor = SuperResolutionProcessor(model=SEN2SRLite, device=device, overlap=32, upscale=4)

for year, config in dataset_configs.items():
    parcels_path = Path(config["parcels_path"])
    log_stage(f"[YEAR {year}] Loading shapefile: {parcels_path.name}...")
    gdf = gpd.read_file(parcels_path)

    for stack_info in config["stacks"]:
        stack_path = Path(stack_info["stack_path"])
        calendar_path = Path(stack_info["calendar_path"])
        log_stage(f"[STACK] Processing Sentinel-2 stack: {stack_path.name}...")

        matcher = TemporalDateMatcher(calendar_path)

        with rasterio.open(stack_path) as src:
            parcels = gdf.to_crs(src.crs)
            parcels = parcels[parcels.intersects(box(*src.bounds))].reset_index(drop=True)
            crs = src.crs

        if parcels.empty:
            log_stage(f"[STACK] No parcels intersect stack: {stack_path.name}. Skipping.")
            continue

        pipeline = ObservationDrivenS2Pipeline(
            stack_path=stack_path,
            parcels=parcels,
            date_matcher=matcher,
            output_dir=out_dir,
            sr_processor=sr_processor,
            crs=crs,
            parcel_buffer_meters=0.0,
            include_advanced_features=True,
            include_texture_features=False,
            include_spatial_features=False,
        )

        lr_inventory, sr_inventory = pipeline.run()
        log_stage(
            f"[DONE] {stack_path.stem}: {len(lr_inventory)} LR rows, {len(sr_inventory)} SR rows, "
            f"combined CSV at {out_dir / 'observations_inventory.csv'}"
        )

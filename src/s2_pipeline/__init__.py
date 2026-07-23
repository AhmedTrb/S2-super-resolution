"""Simple, reusable Sentinel-2 parcel processing pipeline."""

from .date_matcher import TemporalDateMatcher
from .indices import SpectralIndexComputer
from .masks import ParcelMaskGenerator
from .patching import ParcelPatchExtractor, PatchWindow
from .pipeline import ObservationDrivenS2Pipeline, TorchGeoParcelDataset
from .super_resolution import SuperResolutionProcessor
from .yellowness_model import MaskFusionMode, MaskedYellownessRegressor, build_yellowness_model
from .yellowness_training import ParcelYellownessDataset, evaluate_regressor, fit_regressor, make_group_split

__all__ = [
    "TemporalDateMatcher",
    "ParcelPatchExtractor",
    "PatchWindow",
    "ParcelMaskGenerator",
    "SpectralIndexComputer",
    "SuperResolutionProcessor",
    "ObservationDrivenS2Pipeline",
    "TorchGeoParcelDataset",
    "MaskFusionMode",
    "MaskedYellownessRegressor",
    "build_yellowness_model",
    "ParcelYellownessDataset",
    "make_group_split",
    "fit_regressor",
    "evaluate_regressor",
]

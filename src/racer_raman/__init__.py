"""Readout-aware cosmic-ray correction for spectroscopic measurements."""

from .calibration import (
    ImageCalibration,
    SpectrumDarkCalibration,
    TemporalStatistics,
    calibrate_image,
    calibrate_spectrum_dark,
    calibrate_spectrum_reference,
    estimate_temporal_statistics,
)
from .image import (
    ImageConfig,
    ImageModel,
    ParticlePrior,
    build_image_model,
    clean_image,
)
from .spectrum import (
    BayesianSpectrumConfig,
    HybridSpectrumConfig,
    MaskedBaselineConfig,
    SpectrumCalibration,
    build_spectrum_calibration,
    clean_spectrum,
)
from .workflow import CorrectionInfo, ImageCalibrator, SpectrumCalibrator
from .zarr_io import read_image_zarr, read_spectrum_zarr

__all__ = [
    "BayesianSpectrumConfig",
    "CorrectionInfo",
    "HybridSpectrumConfig",
    "ImageCalibration",
    "ImageCalibrator",
    "ImageConfig",
    "ImageModel",
    "MaskedBaselineConfig",
    "ParticlePrior",
    "SpectrumCalibration",
    "SpectrumCalibrator",
    "SpectrumDarkCalibration",
    "TemporalStatistics",
    "build_image_model",
    "build_spectrum_calibration",
    "calibrate_image",
    "calibrate_spectrum_dark",
    "calibrate_spectrum_reference",
    "clean_image",
    "clean_spectrum",
    "estimate_temporal_statistics",
    "read_image_zarr",
    "read_spectrum_zarr",
]

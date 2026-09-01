"""Simple calibration-to-correction workflows for detector measurements."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .calibration import (
    ImageCalibration,
    SpectrumDarkCalibration,
    TemporalStatistics,
    calibrate_image,
    calibrate_spectrum_dark,
    calibrate_spectrum_reference,
)
from .image import ImageConfig, clean_image
from .spectrum import HybridSpectrumConfig, clean_spectrum

__all__ = ["CorrectionInfo", "ImageCalibrator", "SpectrumCalibrator"]


def _is_dataframe(value: object) -> bool:
    return all(
        hasattr(value, attribute)
        for attribute in ("attrs", "columns", "index", "to_numpy")
    )


def _native_array(value: Any) -> np.ndarray:
    converter = getattr(value, "to_numpy", None)
    if callable(converter):
        return np.asarray(converter())
    return np.asarray(value)


def _numeric_array(name: str, value: Any) -> np.ndarray:
    array = _native_array(value)
    if array.dtype.kind not in "uif":
        raise ValueError(f"{name} must contain numeric detector values")
    return array


def _table_columns(value: Any) -> tuple[Any, ...] | None:
    if not _is_dataframe(value):
        return None
    if bool(getattr(value.columns, "has_duplicates", False)):
        raise ValueError("DataFrame detector columns must not contain duplicates")
    return tuple(value.columns.tolist())


def _matching_detector_columns(*values: Any) -> tuple[Any, ...] | None:
    columns = [result for value in values if (result := _table_columns(value))]
    if not columns:
        return None
    expected = columns[0]
    if any(candidate != expected for candidate in columns[1:]):
        raise ValueError("calibration DataFrame detector columns must match")
    return expected


def _validate_matching_image_rows(*values: Any) -> None:
    row_labels = []
    for value in values:
        if not _is_dataframe(value):
            continue
        rows = value.index.get_level_values("row").unique()
        row_labels.append(tuple(rows.tolist()))
    if row_labels and any(candidate != row_labels[0] for candidate in row_labels[1:]):
        raise ValueError("calibration image DataFrame detector rows must match")


def _validate_measurement_columns(
    name: str,
    value: Any,
    expected: tuple[Any, ...] | None,
) -> None:
    columns = _table_columns(value)
    if expected is not None and columns is not None and columns != expected:
        raise ValueError(f"{name} DataFrame detector columns do not match calibration")


def _metadata(value: Any) -> Mapping[str, Any]:
    attributes = getattr(value, "attrs", None)
    return attributes if isinstance(attributes, Mapping) else {}


def _resolved_positive_float(
    name: str,
    explicit: float | None,
    values: tuple[Any, ...],
    *,
    default: float | None,
) -> float:
    candidate: Any = explicit
    if candidate is None:
        inferred = [
            attrs[name] for value in values if name in (attrs := _metadata(value))
        ]
        if inferred:
            converted = [float(value) for value in inferred]
            if any(value != converted[0] for value in converted[1:]):
                raise ValueError(f"calibration DataFrame {name} attributes must match")
            candidate = converted[0]
        else:
            candidate = default
    if candidate is None:
        raise ValueError(
            f"{name} is required when calibration data have no {name} attribute"
        )
    result = float(candidate)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _image_stack(name: str, value: Any) -> np.ndarray:
    array = _numeric_array(name, value)
    if not _is_dataframe(value):
        return array

    index = value.index
    if getattr(index, "nlevels", None) != 2 or tuple(index.names) != (
        "frame",
        "row",
    ):
        raise ValueError(f"{name} DataFrame index must be a (frame, row) MultiIndex")
    if bool(getattr(index, "has_duplicates", False)):
        raise ValueError(f"{name} DataFrame index must not contain duplicates")

    frame_level = index.get_level_values("frame")
    row_level = index.get_level_values("row")
    frame_ids = np.asarray(frame_level.unique())
    row_ids = np.asarray(row_level.unique())
    expected_size = frame_ids.size * row_ids.size
    if frame_ids.size == 0 or row_ids.size == 0 or expected_size != len(index):
        raise ValueError(f"{name} DataFrame must contain a complete frame-row grid")
    expected_frames = np.repeat(frame_ids, row_ids.size)
    expected_rows = np.tile(row_ids, frame_ids.size)
    frames_are_ordered = np.array_equal(np.asarray(frame_level), expected_frames)
    rows_are_ordered = np.array_equal(np.asarray(row_level), expected_rows)
    if not frames_are_ordered or not rows_are_ordered:
        raise ValueError(
            f"{name} DataFrame rows must be ordered as a complete (frame, row) grid"
        )
    return array.reshape(frame_ids.size, row_ids.size, array.shape[1])


def _spectrum_stack(name: str, value: Any) -> np.ndarray:
    array = _numeric_array(name, value)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (frames, detector columns)")
    return array


def _spectrum_measurement(
    name: str,
    value: Any,
    expected_columns: tuple[Any, ...] | None,
) -> np.ndarray:
    array = _numeric_array(name, value)
    if _is_dataframe(value):
        if array.ndim != 2 or 1 not in array.shape:
            raise ValueError(f"{name} DataFrame must contain exactly one spectrum")
        if array.shape[0] == 1:
            labels = _table_columns(value)
            result = array[0]
        else:
            if bool(getattr(value.index, "has_duplicates", False)):
                raise ValueError(f"{name} DataFrame detector index must be unique")
            labels = tuple(value.index.tolist())
            result = array[:, 0]
        if expected_columns is not None and labels != expected_columns:
            raise ValueError(f"{name} detector columns do not match calibration")
        return result

    if array.ndim == 1 and expected_columns is not None and hasattr(value, "index"):
        labels = tuple(value.index.tolist())
        if labels != expected_columns:
            raise ValueError(f"{name} detector columns do not match calibration")
    return array


@dataclass(frozen=True, slots=True)
class CorrectionInfo:
    """Small summary of the values changed by one correction."""

    corrected_mask: np.ndarray
    removed_charge_adu: float

    def __post_init__(self) -> None:
        mask = np.array(self.corrected_mask, dtype=bool, copy=True)
        removed = float(self.removed_charge_adu)
        if not np.isfinite(removed):
            raise ValueError("removed_charge_adu must be finite")
        mask.setflags(write=False)
        object.__setattr__(self, "corrected_mask", mask)
        object.__setattr__(self, "removed_charge_adu", removed)

    @property
    def event_found(self) -> bool:
        """Whether the correction changed at least one detector value."""

        return bool(np.any(self.corrected_mask))

    @property
    def n_corrected(self) -> int:
        """Number of detector values changed by the correction."""

        return int(np.count_nonzero(self.corrected_mask))


@dataclass(frozen=True, slots=True)
class ImageCalibrator:
    """Fit once from image stacks and correct raw detector images."""

    _calibration: ImageCalibration = field(repr=False)
    _config: ImageConfig = field(repr=False)
    _detector_columns: tuple[Any, ...] | None = field(repr=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self._calibration, ImageCalibration):
            raise TypeError("_calibration must be ImageCalibration")
        if not isinstance(self._config, ImageConfig):
            raise TypeError("_config must be ImageConfig")

    @classmethod
    def fit(
        cls,
        dark_frames_adu: Any,
        sample_frames_adu: Any,
        *,
        gain_e_per_adu: float | None = None,
        saturation_adu: float | None = None,
    ) -> ImageCalibrator:
        """Fit an image corrector from dark and open-shutter sample frames."""

        detector_columns = _matching_detector_columns(
            dark_frames_adu,
            sample_frames_adu,
        )
        gain = _resolved_positive_float(
            "gain_e_per_adu",
            gain_e_per_adu,
            (dark_frames_adu, sample_frames_adu),
            default=0.7,
        )
        saturation = _resolved_positive_float(
            "saturation_adu",
            saturation_adu,
            (dark_frames_adu, sample_frames_adu),
            default=65535.0,
        )
        dark = _image_stack("dark_frames_adu", dark_frames_adu)
        sample = _image_stack("sample_frames_adu", sample_frames_adu)
        _validate_matching_image_rows(dark_frames_adu, sample_frames_adu)
        calibration = calibrate_image(
            dark,
            sample,
            gain_e_per_adu=gain,
            saturation_adu=saturation,
        )
        config = ImageConfig(
            gain_e_per_adu=gain,
            saturation_adu=saturation,
        )
        config.validate()
        return cls(calibration, config, detector_columns)

    def correct(
        self,
        raw_image_adu: Any,
        *,
        return_info: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, CorrectionInfo]:
        """Return one corrected image in the same raw ADU scale as the input."""

        _validate_measurement_columns(
            "raw_image_adu",
            raw_image_adu,
            self._detector_columns,
        )
        raw = _numeric_array("raw_image_adu", raw_image_adu)
        if raw.ndim != 2:
            raise ValueError("raw_image_adu must be a 2-D detector image")
        if np.any(~np.isfinite(raw)):
            raise ValueError("raw_image_adu must be finite")
        raw64 = np.asarray(raw, dtype=np.float64)
        cleaned64 = clean_image(
            raw64,
            self._calibration.background_adu,
            self._calibration.model,
            config=self._config,
        )
        changed = np.asarray(cleaned64 != raw64, dtype=bool)
        if np.any(changed):
            corrected = raw64.copy()
            corrected[changed] = cleaned64[changed]
            removed = float(np.sum(raw64[changed] - corrected[changed]))
        else:
            corrected = np.array(raw, copy=True)
            removed = 0.0
        if not return_info:
            return corrected
        return corrected, CorrectionInfo(changed, removed)


@dataclass(frozen=True, slots=True)
class SpectrumCalibrator:
    """Fit once from spectrum stacks and correct raw row-summed spectra."""

    _dark: SpectrumDarkCalibration = field(repr=False)
    _reference: TemporalStatistics = field(repr=False)
    _saturation_adu: float = field(repr=False)
    _config: HybridSpectrumConfig = field(repr=False)
    _detector_columns: tuple[Any, ...] | None = field(repr=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self._dark, SpectrumDarkCalibration):
            raise TypeError("_dark must be SpectrumDarkCalibration")
        if not isinstance(self._reference, TemporalStatistics):
            raise TypeError("_reference must be TemporalStatistics")
        if not isinstance(self._config, HybridSpectrumConfig):
            raise TypeError("_config must be HybridSpectrumConfig")
        saturation = float(self._saturation_adu)
        if not np.isfinite(saturation) or saturation <= 0.0:
            raise ValueError("_saturation_adu must be finite and positive")
        object.__setattr__(self, "_saturation_adu", saturation)

    @classmethod
    def fit(
        cls,
        dark_frames_adu: Any,
        reference_frames_adu: Any,
        *,
        exposure_ms: float | None = None,
        saturation_adu: float | None = None,
    ) -> SpectrumCalibrator:
        """Fit a spectrum corrector from dark and open-reference frames."""

        detector_columns = _matching_detector_columns(
            dark_frames_adu,
            reference_frames_adu,
        )
        exposure = _resolved_positive_float(
            "exposure_ms",
            exposure_ms,
            (dark_frames_adu, reference_frames_adu),
            default=None,
        )
        saturation = _resolved_positive_float(
            "saturation_adu",
            saturation_adu,
            (dark_frames_adu, reference_frames_adu),
            default=65535.0,
        )
        dark_frames = _spectrum_stack("dark_frames_adu", dark_frames_adu)
        reference_frames = _spectrum_stack(
            "reference_frames_adu",
            reference_frames_adu,
        )
        dark = calibrate_spectrum_dark(
            dark_frames,
            exposure_ms=exposure,
            saturation_adu=saturation,
        )
        reference = calibrate_spectrum_reference(
            reference_frames,
            dark,
            saturation_adu=saturation,
        )
        return cls(
            dark,
            reference,
            saturation,
            HybridSpectrumConfig(),
            detector_columns,
        )

    def correct(
        self,
        raw_spectrum_adu: Any,
        *,
        return_info: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, CorrectionInfo]:
        """Return one corrected spectrum in the raw input ADU scale."""

        raw = _spectrum_measurement(
            "raw_spectrum_adu",
            raw_spectrum_adu,
            self._detector_columns,
        )
        if raw.ndim != 1:
            raise ValueError("raw_spectrum_adu must be a 1-D spectrum")
        if raw.size != self._reference.master_adu.size:
            raise ValueError("raw_spectrum_adu size does not match calibration")
        if np.any(~np.isfinite(raw)):
            raise ValueError("raw_spectrum_adu must be finite")
        if np.any(raw >= self._saturation_adu):
            raise ValueError("raw_spectrum_adu must be below saturation_adu")

        raw64 = np.asarray(raw, dtype=np.float64)
        frame_offset = float(np.median(raw64 - self._reference.master_adu))
        work = raw64 - self._dark.master_adu - frame_offset
        cleaned_work = clean_spectrum(
            work,
            self._reference.variance_adu2,
            self._dark.particle_calibration,
            config=self._config,
        )
        changed = np.asarray(cleaned_work != work, dtype=bool)
        if np.any(changed):
            corrected = raw64.copy()
            corrected[changed] += cleaned_work[changed] - work[changed]
            removed = float(np.sum(raw64[changed] - corrected[changed]))
        else:
            corrected = np.array(raw, copy=True)
            removed = 0.0
        if not return_info:
            return corrected
        return corrected, CorrectionInfo(changed, removed)

    def correct_many(
        self,
        raw_spectra_adu: Any,
        *,
        return_info: bool = False,
    ) -> Any:
        """Correct a stack of spectra while preserving a DataFrame layout."""

        _validate_measurement_columns(
            "raw_spectra_adu",
            raw_spectra_adu,
            self._detector_columns,
        )
        raw = _numeric_array("raw_spectra_adu", raw_spectra_adu)
        if raw.ndim != 2:
            raise ValueError(
                "raw_spectra_adu must have shape (spectra, detector columns)"
            )
        if raw.shape[0] == 0:
            raise ValueError("raw_spectra_adu must contain at least one spectrum")
        if raw.shape[1] != self._reference.master_adu.size:
            raise ValueError(
                "raw_spectra_adu detector columns do not match calibration"
            )
        if np.any(~np.isfinite(raw)):
            raise ValueError("raw_spectra_adu must be finite")
        if np.any(raw >= self._saturation_adu):
            raise ValueError("raw_spectra_adu must be below saturation_adu")

        corrected_rows = []
        information = []
        for spectrum in raw:
            corrected, info = self.correct(spectrum, return_info=True)
            corrected_rows.append(corrected)
            information.append(info)

        summaries = tuple(information)
        if any(info.event_found for info in summaries):
            corrected_values = np.stack(corrected_rows)
            if _is_dataframe(raw_spectra_adu):
                corrected_batch = type(raw_spectra_adu)(
                    corrected_values,
                    index=raw_spectra_adu.index.copy(),
                    columns=raw_spectra_adu.columns.copy(),
                )
            else:
                corrected_batch = corrected_values
        elif _is_dataframe(raw_spectra_adu):
            corrected_batch = raw_spectra_adu.copy(deep=True)
        else:
            corrected_batch = np.array(raw, copy=True)

        if _is_dataframe(raw_spectra_adu):
            corrected_batch.attrs.clear()
            corrected_batch.attrs.update(deepcopy(dict(_metadata(raw_spectra_adu))))

        if not return_info:
            return corrected_batch
        return corrected_batch, summaries

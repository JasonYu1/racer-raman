"""Array-only calibration from independent dark and sample frame stacks."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .image import ImageModel, ParticlePrior, build_image_model
from .spectrum import SpectrumCalibration, build_spectrum_calibration

__all__ = [
    "ImageCalibration",
    "SpectrumDarkCalibration",
    "TemporalStatistics",
    "calibrate_image",
    "calibrate_spectrum_dark",
    "calibrate_spectrum_reference",
    "estimate_temporal_statistics",
]


@dataclass(frozen=True)
class ImageCalibration:
    """Calibrated background, image model, and dark-frame diagnostics."""

    background_adu: np.ndarray
    model: ImageModel
    dark_master_adu: np.ndarray
    dark_variance_adu2: np.ndarray
    dark_frame_offsets_adu: np.ndarray
    n_dark_events: int
    n_dark_event_pixels: int

    def __post_init__(self) -> None:
        background = np.array(self.background_adu, dtype=np.float64, copy=True)
        master = np.array(self.dark_master_adu, dtype=np.float64, copy=True)
        variance = np.array(self.dark_variance_adu2, dtype=np.float64, copy=True)
        offsets = np.array(
            self.dark_frame_offsets_adu,
            dtype=np.float64,
            copy=True,
        )
        if not isinstance(self.model, ImageModel):
            raise TypeError("model must be ImageModel")
        if background.ndim != 2 or background.shape != self.model.profile.shape:
            raise ValueError("background_adu must match the image model")
        if master.shape != background.shape or variance.shape != background.shape:
            raise ValueError("dark master and variance must match background_adu")
        if np.any(~np.isfinite(background)) or np.any(~np.isfinite(master)):
            raise ValueError("background and dark master must be finite")
        if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
            raise ValueError("dark variance must be finite and positive")
        if offsets.ndim != 1 or offsets.size < 3 or np.any(~np.isfinite(offsets)):
            raise ValueError("dark frame offsets must contain at least three values")
        event_count = _nonnegative_integer("n_dark_events", self.n_dark_events)
        event_pixels = _nonnegative_integer(
            "n_dark_event_pixels",
            self.n_dark_event_pixels,
        )
        if event_pixels < event_count:
            raise ValueError("n_dark_event_pixels cannot be smaller than n_dark_events")
        for array in (background, master, variance, offsets):
            array.setflags(write=False)
        object.__setattr__(self, "background_adu", background)
        object.__setattr__(self, "dark_master_adu", master)
        object.__setattr__(self, "dark_variance_adu2", variance)
        object.__setattr__(self, "dark_frame_offsets_adu", offsets)
        object.__setattr__(self, "n_dark_events", event_count)
        object.__setattr__(self, "n_dark_event_pixels", event_pixels)


@dataclass(frozen=True)
class TemporalStatistics:
    """Robust per-column temporal location and variance."""

    master_adu: np.ndarray
    variance_adu2: np.ndarray
    frame_offsets_adu: np.ndarray
    valid_counts: np.ndarray

    def __post_init__(self) -> None:
        master = np.array(self.master_adu, dtype=np.float64, copy=True)
        variance = np.array(self.variance_adu2, dtype=np.float64, copy=True)
        offsets = np.array(self.frame_offsets_adu, dtype=np.float64, copy=True)
        counts_raw = np.asarray(self.valid_counts)
        if master.ndim != 1 or master.size == 0:
            raise ValueError("master_adu must be a nonempty 1-D array")
        if variance.shape != master.shape:
            raise ValueError("variance_adu2 must match master_adu")
        if np.any(~np.isfinite(master)):
            raise ValueError("master_adu must be finite")
        if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
            raise ValueError("variance_adu2 must be finite and positive")
        if offsets.ndim != 1 or offsets.size < 3:
            raise ValueError("frame_offsets_adu must contain at least three frames")
        if np.any(~np.isfinite(offsets)):
            raise ValueError("frame_offsets_adu must be finite")
        if counts_raw.shape != master.shape or counts_raw.dtype.kind not in "uif":
            raise ValueError("valid_counts must be numeric and match master_adu")
        counts_float = np.asarray(counts_raw, dtype=np.float64)
        if (
            np.any(~np.isfinite(counts_float))
            or np.any(counts_float < 0)
            or np.any(counts_float != np.floor(counts_float))
            or np.any(counts_float > offsets.size)
        ):
            raise ValueError(
                "valid_counts must contain whole counts between zero and frame count"
            )
        counts = counts_float.astype(np.int32)
        for array in (master, variance, offsets, counts):
            array.setflags(write=False)
        object.__setattr__(self, "master_adu", master)
        object.__setattr__(self, "variance_adu2", variance)
        object.__setattr__(self, "frame_offsets_adu", offsets)
        object.__setattr__(self, "valid_counts", counts)


@dataclass(frozen=True)
class SpectrumDarkCalibration:
    """Closed-shutter temporal statistics and empirical particle prior."""

    temporal: TemporalStatistics
    particle_calibration: SpectrumCalibration

    def __post_init__(self) -> None:
        if not isinstance(self.temporal, TemporalStatistics):
            raise TypeError("temporal must be TemporalStatistics")
        if not isinstance(self.particle_calibration, SpectrumCalibration):
            raise TypeError("particle_calibration must be SpectrumCalibration")
        if self.temporal.master_adu.size < 1:
            raise ValueError("dark calibration must contain detector columns")

    @property
    def master_adu(self) -> np.ndarray:
        """Per-column closed-shutter level."""

        return self.temporal.master_adu

    @property
    def variance_adu2(self) -> np.ndarray:
        """Per-column closed-shutter variance."""

        return self.temporal.variance_adu2


def _validated_stack(stack_adu: np.ndarray) -> np.ndarray:
    raw = np.asarray(stack_adu)
    if raw.ndim != 2 or raw.shape[0] < 3 or raw.shape[1] < 1:
        raise ValueError(
            "stack_adu must have shape (at least three frames, detector columns)"
        )
    if raw.dtype.kind not in "uif":
        raise ValueError("stack_adu must contain numeric detector values")
    return np.asarray(raw, dtype=np.float64)


def _positive_float(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _validated_image_stack(name: str, stack_adu: np.ndarray) -> np.ndarray:
    raw = np.asarray(stack_adu)
    if raw.ndim != 3 or raw.shape[0] < 3 or min(raw.shape[1:]) < 1:
        raise ValueError(
            f"{name} must have shape "
            "(at least three frames, detector rows, detector columns)"
        )
    if raw.dtype.kind not in "uif":
        raise ValueError(f"{name} must contain numeric detector values")
    data = np.asarray(raw, dtype=np.float64)
    if np.any(~np.isfinite(data)):
        raise ValueError(f"{name} must contain finite detector values")
    return data


def _image_dark_statistics(
    dark_frames_adu: np.ndarray,
    *,
    mad_to_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pilot = np.median(dark_frames_adu, axis=0)
    offsets = np.median(
        dark_frames_adu - pilot[None, :, :],
        axis=(1, 2),
    )
    aligned = dark_frames_adu - offsets[:, None, None]
    master = np.median(aligned, axis=0)
    residual = aligned - master[None, :, :]
    sigma = mad_to_sigma * np.median(np.abs(residual), axis=0)
    variance = np.square(sigma)
    positive = variance[np.isfinite(variance) & (variance > 0)]
    if positive.size == 0:
        raise ValueError("dark frames do not contain measurable temporal variance")
    variance = np.maximum(variance, float(np.median(positive)))
    return master, variance, offsets, residual


def _image_dark_events(
    dark_frames_adu: np.ndarray,
    residual_adu: np.ndarray,
    variance_adu2: np.ndarray,
    *,
    saturation_adu: float,
    seed_sigma: float,
    growth_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zscore = residual_adu / np.sqrt(variance_adu2)[None, :, :]
    zscore[dark_frames_adu >= saturation_adu] = -np.inf
    event_charges: list[float] = []
    event_sizes: list[int] = []
    event_rows: list[int] = []
    connectivity = np.ones((3, 3), dtype=bool)
    for frame in range(dark_frames_adu.shape[0]):
        labels, _ = ndi.label(zscore[frame] >= growth_sigma, connectivity)
        seed_labels = np.unique(labels[zscore[frame] >= seed_sigma])
        for label in seed_labels:
            if label == 0:
                continue
            component = labels == label
            rows, _ = np.nonzero(component)
            charges = residual_adu[frame, component]
            event_charges.extend(charges.tolist())
            event_rows.extend(rows.tolist())
            event_sizes.append(int(charges.size))
    return (
        np.sort(np.asarray(event_charges, dtype=np.float64)),
        np.asarray(event_sizes, dtype=np.int32),
        np.asarray(event_rows, dtype=np.int32),
    )


def _aligned_image_template(
    stack_adu: np.ndarray,
    dark_master_adu: np.ndarray,
) -> np.ndarray:
    pilot = np.median(stack_adu, axis=0)
    offsets = np.median(
        stack_adu - pilot[None, :, :],
        axis=(1, 2),
    )
    common_offset = float(np.median(offsets))
    aligned = stack_adu - (offsets - common_offset)[:, None, None]
    return np.median(aligned, axis=0) - dark_master_adu


def calibrate_image(
    dark_frames_adu: np.ndarray,
    sample_profile_frames_adu: np.ndarray,
    *,
    gain_e_per_adu: float = 0.7,
    mad_to_sigma: float = 1.4826,
    saturation_adu: float = 65535.0,
    event_seed_sigma: float = 8.0,
    event_grow_sigma: float = 3.0,
    trace_half_width_rows: int = 12,
    profile_smoothing_width_columns: int = 9,
    ridge_smoothing_width_columns: int = 15,
    row_smoothing_width: int = 9,
    amplitude_components: int = 4,
) -> ImageCalibration:
    """Build a 2-D model from closed-shutter and sample training stacks."""

    dark = _validated_image_stack("dark_frames_adu", dark_frames_adu)
    sample = _validated_image_stack(
        "sample_profile_frames_adu",
        sample_profile_frames_adu,
    )
    if sample.shape[1:] != dark.shape[1:]:
        raise ValueError("dark and sample-profile images must match")
    gain = _positive_float("gain_e_per_adu", gain_e_per_adu)
    mad_scale = _positive_float("mad_to_sigma", mad_to_sigma)
    saturation = _positive_float("saturation_adu", saturation_adu)
    seed_threshold = _positive_float("event_seed_sigma", event_seed_sigma)
    growth_threshold = _positive_float("event_grow_sigma", event_grow_sigma)
    if seed_threshold <= growth_threshold:
        raise ValueError("event_seed_sigma must be greater than event_grow_sigma")
    half_width = _nonnegative_integer(
        "trace_half_width_rows",
        trace_half_width_rows,
    )
    if 2 * half_width + 1 >= dark.shape[1]:
        raise ValueError("trace width must leave detector rows outside the trace")
    profile_width = _positive_integer(
        "profile_smoothing_width_columns",
        profile_smoothing_width_columns,
    )
    ridge_width = _positive_integer(
        "ridge_smoothing_width_columns",
        ridge_smoothing_width_columns,
    )
    row_width = _positive_integer("row_smoothing_width", row_smoothing_width)
    component_count = _positive_integer("amplitude_components", amplitude_components)

    master, dark_variance, dark_offsets, dark_residual = _image_dark_statistics(
        dark,
        mad_to_sigma=mad_scale,
    )
    charges, event_sizes, event_rows = _image_dark_events(
        dark,
        dark_residual,
        dark_variance,
        saturation_adu=saturation,
        seed_sigma=seed_threshold,
        growth_sigma=growth_threshold,
    )
    if event_sizes.size == 0:
        raise ValueError("closed-shutter stack contains no detected particle events")
    if charges.size < component_count:
        raise ValueError("too few event pixels for amplitude_components")
    charge_groups = np.array_split(charges, component_count)
    amplitude_scales = np.asarray(
        [float(np.mean(group)) for group in charge_groups],
        dtype=np.float64,
    )
    amplitude_weights = np.asarray(
        [group.size for group in charge_groups],
        dtype=np.float64,
    )
    amplitude_weights /= charges.size
    row_counts = np.bincount(event_rows, minlength=dark.shape[1]).astype(np.float64)
    row_counts = ndi.uniform_filter1d(
        row_counts,
        size=row_width,
        mode="nearest",
    )
    row_probability = (row_counts + 0.5) / (
        dark.shape[0] * dark.shape[2] + 0.5 * dark.shape[1]
    )
    particle_prior = ParticlePrior(
        row_probability=row_probability,
        amplitude_scales_adu=amplitude_scales,
        amplitude_weights=amplitude_weights,
        maximum_event_pixels=int(np.max(event_sizes)),
    )

    sample_template = _aligned_image_template(sample, master)
    sample_floor = np.percentile(sample_template, 20.0, axis=0)
    measured_sample = np.clip(sample_template - sample_floor[None, :], 0.0, None)
    measured_sample = ndi.uniform_filter1d(
        measured_sample,
        size=profile_width,
        axis=1,
        mode="nearest",
    )
    row_signal = ndi.uniform_filter1d(
        measured_sample.sum(axis=1),
        size=row_width,
        mode="nearest",
    )
    trace_center = int(np.argmax(row_signal))
    detector_rows = np.arange(dark.shape[1], dtype=np.float64)[:, None]
    centroid_window = np.abs(detector_rows - trace_center) <= half_width
    centroid_weights = measured_sample * centroid_window
    centroid_normalization = centroid_weights.sum(axis=0)
    if np.any(centroid_normalization <= 0):
        raise ValueError("sample stack must contain a trace in every detector column")
    ridge = np.rint(
        np.sum(detector_rows * centroid_weights, axis=0) / centroid_normalization
    ).astype(np.intp)
    ridge = ndi.median_filter(ridge, size=ridge_width, mode="nearest")
    trace_mask = np.abs(detector_rows - ridge[None, :]) <= half_width
    sample_trace = measured_sample * trace_mask
    sample_normalization = sample_trace.sum(axis=0)
    measured_sample_profile = np.divide(
        sample_trace,
        sample_normalization[None, :],
        out=np.zeros_like(sample_trace),
        where=sample_normalization[None, :] > 0,
    )

    diffuse = np.empty(dark.shape[2], dtype=np.float64)
    for column in range(dark.shape[2]):
        off_trace = sample_template[~trace_mask[:, column], column]
        diffuse[column] = float(np.median(off_trace))
    diffuse = ndi.median_filter(diffuse, size=profile_width, mode="nearest")
    optical_template = (
        np.clip(
            sample_template - diffuse[None, :],
            0.0,
            None,
        )
        * trace_mask
    )
    optical_template = ndi.uniform_filter1d(
        optical_template,
        size=profile_width,
        axis=1,
        mode="nearest",
    )
    optical_template *= trace_mask
    normalization = optical_template.sum(axis=0)
    profile = np.divide(
        optical_template,
        normalization[None, :],
        out=measured_sample_profile.copy(),
        where=normalization[None, :] > 0,
    )
    detector_variance = dark_variance + np.clip(master, 0.0, None) / gain
    numerator = np.sum(profile * optical_template / detector_variance, axis=0)
    denominator = np.sum(profile * profile / detector_variance, axis=0)
    template_spectrum = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    model = build_image_model(
        optical_template,
        trace_mask,
        particle_prior,
        detector_variance_adu2=detector_variance,
        spatial_profile_template=profile,
        template_spectrum_adu=template_spectrum,
        relative_profile_floor=0.0,
    )
    return ImageCalibration(
        background_adu=master + diffuse[None, :],
        model=model,
        dark_master_adu=master,
        dark_variance_adu2=dark_variance,
        dark_frame_offsets_adu=dark_offsets,
        n_dark_events=int(event_sizes.size),
        n_dark_event_pixels=int(charges.size),
    )


def _nanmedian(values: np.ndarray, *, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def _fill_missing(
    values: np.ndarray,
    fallback: np.ndarray | None,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    missing = ~np.isfinite(result)
    if fallback is not None and np.any(missing):
        fallback_values = np.asarray(fallback, dtype=np.float64)
        result[missing] = fallback_values[missing]
    missing = ~np.isfinite(result)
    if np.any(missing):
        finite = np.flatnonzero(np.isfinite(result))
        if finite.size == 0:
            raise ValueError("stack_adu contains no usable detector values")
        locations = np.arange(result.size)
        result[missing] = np.interp(locations[missing], finite, result[finite])
    return result


def estimate_temporal_statistics(
    stack_adu: np.ndarray,
    *,
    fallback: TemporalStatistics | None = None,
    saturation_adu: float = 65535.0,
    iterations: int = 2,
    minimum_sigma_adu: float = 0.5,
) -> TemporalStatistics:
    """Estimate a robust temporal master and per-column variance.

    Per-frame common offsets are removed iteratively. A median absolute
    deviation supplies a six-sigma particle clip, after which the variance is
    the clipped second moment of all usable frames.
    """

    data = _validated_stack(stack_adu)
    saturation = _positive_float("saturation_adu", saturation_adu)
    iteration_count = _positive_integer("iterations", iterations)
    minimum_sigma = _positive_float("minimum_sigma_adu", minimum_sigma_adu)
    if fallback is not None and not isinstance(fallback, TemporalStatistics):
        raise TypeError("fallback must be TemporalStatistics or None")
    columns = data.shape[1]
    if fallback is not None and fallback.master_adu.size != columns:
        raise ValueError("fallback detector columns must match stack_adu")

    fallback_master = None if fallback is None else fallback.master_adu
    fallback_variance = None if fallback is None else fallback.variance_adu2
    valid = np.isfinite(data) & (data < saturation)
    masked = np.where(valid, data, np.nan)
    master = _fill_missing(_nanmedian(masked, axis=0), fallback_master)
    frame_offsets = np.zeros(data.shape[0], dtype=np.float64)

    for _ in range(iteration_count):
        residual = np.where(valid, data - master[None, :], np.nan)
        frame_offsets = _nanmedian(residual, axis=1)
        frame_offsets[~np.isfinite(frame_offsets)] = 0.0
        centered = np.where(valid, data - frame_offsets[:, None], np.nan)
        master = _fill_missing(_nanmedian(centered, axis=0), fallback_master)

    residual = np.where(
        valid,
        data - master[None, :] - frame_offsets[:, None],
        np.nan,
    )
    residual_location = _nanmedian(residual, axis=0)
    centered_residual = residual - residual_location[None, :]
    initial_sigma = 1.4826 * _nanmedian(np.abs(centered_residual), axis=0)
    valid_counts = np.sum(valid, axis=0).astype(np.int32)

    if fallback_variance is None:
        positive = initial_sigma[np.isfinite(initial_sigma) & (initial_sigma > 0)]
        global_sigma = float(np.median(positive)) if positive.size else minimum_sigma
        fallback_sigma = np.full(columns, global_sigma, dtype=np.float64)
    else:
        fallback_sigma = np.sqrt(fallback_variance)
    unusable = (~np.isfinite(initial_sigma)) | (initial_sigma <= 0) | (valid_counts < 3)
    initial_sigma[unusable] = fallback_sigma[unusable]
    initial_sigma = np.maximum(initial_sigma, minimum_sigma)

    inlier = valid & (np.abs(centered_residual) < 6.0 * initial_sigma[None, :])
    inlier_count = np.sum(inlier, axis=0)
    variance = np.divide(
        np.nansum(
            np.where(
                inlier,
                centered_residual * centered_residual,
                np.nan,
            ),
            axis=0,
        ),
        inlier_count,
        out=np.square(initial_sigma),
        where=inlier_count > 2,
    )
    sigma = np.sqrt(np.maximum(variance, 0.0))
    bad_sigma = (~np.isfinite(sigma)) | (sigma <= 0)
    positive_sigma = sigma[~bad_sigma]
    global_sigma = (
        float(np.median(positive_sigma)) if positive_sigma.size else minimum_sigma
    )
    sigma[bad_sigma] = global_sigma
    sigma = np.maximum(sigma, minimum_sigma)
    variance = np.square(sigma)
    return TemporalStatistics(
        master_adu=master,
        variance_adu2=variance,
        frame_offsets_adu=frame_offsets,
        valid_counts=valid_counts,
    )


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def _dark_events(
    stack_adu: np.ndarray,
    temporal: TemporalStatistics,
    *,
    saturation_adu: float,
    seed_sigma: float,
    growth_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(stack_adu) & (stack_adu < saturation_adu)
    residual = (
        stack_adu - temporal.master_adu[None, :] - temporal.frame_offsets_adu[:, None]
    )
    standardized = residual / np.sqrt(temporal.variance_adu2)[None, :]
    standardized[~valid] = -np.inf
    charge_samples: list[float] = []
    widths: list[int] = []
    for frame in range(stack_adu.shape[0]):
        seed = standardized[frame] >= seed_sigma
        growth = standardized[frame] >= growth_sigma
        for start, stop in _true_runs(growth):
            if not np.any(seed[start : stop + 1]):
                continue
            event_charge = np.clip(residual[frame, start : stop + 1], 0.0, None)
            charge_samples.extend(event_charge.tolist())
            widths.append(stop - start + 1)
    return (
        np.asarray(charge_samples, dtype=np.float64),
        np.asarray(widths, dtype=np.int32),
    )


def calibrate_spectrum_dark(
    stack_adu: np.ndarray,
    *,
    exposure_ms: float,
    saturation_adu: float = 65535.0,
    temporal_iterations: int = 2,
    minimum_sigma_adu: float = 0.5,
    seed_sigma: float = 8.0,
    growth_sigma: float = 3.0,
    maximum_samples: int = 4000,
    seed: int = 20260819,
) -> SpectrumDarkCalibration:
    """Calibrate one row-summed exposure from raw closed-shutter frames."""

    data = _validated_stack(stack_adu)
    saturation = _positive_float("saturation_adu", saturation_adu)
    seed_threshold = _positive_float("seed_sigma", seed_sigma)
    growth_threshold = _positive_float("growth_sigma", growth_sigma)
    if seed_threshold <= growth_threshold:
        raise ValueError("seed_sigma must be greater than growth_sigma")
    temporal = estimate_temporal_statistics(
        data,
        saturation_adu=saturation,
        iterations=temporal_iterations,
        minimum_sigma_adu=minimum_sigma_adu,
    )
    charge_samples, widths = _dark_events(
        data,
        temporal,
        saturation_adu=saturation,
        seed_sigma=seed_threshold,
        growth_sigma=growth_threshold,
    )
    if widths.size == 0:
        raise ValueError("closed-shutter stack contains no detected particle events")
    particle = build_spectrum_calibration(
        charge_samples,
        widths,
        events_per_frame=widths.size / data.shape[0],
        detector_columns=data.shape[1],
        exposure_ms=exposure_ms,
        maximum_samples=maximum_samples,
        seed=seed,
    )
    return SpectrumDarkCalibration(
        temporal=temporal,
        particle_calibration=particle,
    )


def calibrate_spectrum_reference(
    stack_adu: np.ndarray,
    dark: SpectrumDarkCalibration,
    *,
    saturation_adu: float = 65535.0,
    temporal_iterations: int = 2,
    minimum_sigma_adu: float = 0.5,
) -> TemporalStatistics:
    """Measure an independent open-reference stack with dark fallbacks."""

    if not isinstance(dark, SpectrumDarkCalibration):
        raise TypeError("dark must be SpectrumDarkCalibration")
    reference = estimate_temporal_statistics(
        stack_adu,
        fallback=dark.temporal,
        saturation_adu=saturation_adu,
        iterations=temporal_iterations,
        minimum_sigma_adu=minimum_sigma_adu,
    )
    return TemporalStatistics(
        master_adu=reference.master_adu,
        variance_adu2=np.maximum(reference.variance_adu2, dark.variance_adu2),
        frame_offsets_adu=reference.frame_offsets_adu,
        valid_counts=reference.valid_counts,
    )

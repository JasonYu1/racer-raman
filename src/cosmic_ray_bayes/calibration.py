"""Array-only calibration from independent dark and sample frame stacks."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .charge_prior import select_charge_prior
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
    # b_x of Algorithm S1a step 10: the off-trace diffuse photo level, one
    # value per column. A calibrated output in its own right, and distinct
    # from background_adu, which is D + B^inst.
    diffuse_photo_adu: np.ndarray
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


def _evenly_spaced_indices(n_frames: int, maximum: int) -> np.ndarray:
    """Pick at most ``maximum`` evenly spaced frame indices, odd where possible.

    An odd count keeps every pilot median an actual sample rather than the
    mean of two, which matters because the master dark is a median.
    """

    count = min(int(n_frames), int(maximum))
    if count > 1 and count % 2 == 0:
        count -= 1
    return np.unique(np.linspace(0, int(n_frames) - 1, count, dtype=int))


def _grid_offsets(
    frames_adu: np.ndarray,
    reference_adu: np.ndarray,
    *,
    stride: int,
) -> np.ndarray:
    """Robust scalar offset per frame, from a spatially thinned pixel grid."""

    thinned = frames_adu[:, ::stride, ::stride] - reference_adu[::stride, ::stride]
    return np.median(thinned, axis=(1, 2))


def _image_dark_statistics(
    dark_frames_adu: np.ndarray,
    *,
    mad_to_sigma: float,
    pilot_frames: int,
    offset_stride: int,
    variance_clip_sigma: float,
    minimum_sigma_adu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Master dark, detector variance, frame offsets, residuals and scale.

    Implements Algorithm S1a steps 1.1-1.9 of the supplement.  The pilot
    frames supply the master dark and the per-pixel scale; the full stack
    supplies the variance and the event catalogue.  Everything is in ADU.
    """

    n_frames = int(dark_frames_adu.shape[0])
    pilot_index = _evenly_spaced_indices(n_frames, pilot_frames)
    pilot = dark_frames_adu[pilot_index]

    # S1a.1-1.3  preliminary dark, pilot offsets on the thinned grid, master.
    reference = np.median(pilot, axis=0)
    pilot_offsets = _grid_offsets(pilot, reference, stride=offset_stride)
    pilot = pilot - pilot_offsets[:, None, None]
    master = np.median(pilot, axis=0)

    # S1a.4  per-pixel scale from the pilot residuals, inner median removed.
    pilot_residual = pilot - master[None, :, :]
    centre = np.median(pilot_residual, axis=0, keepdims=True)
    sigma = mad_to_sigma * np.median(np.abs(pilot_residual - centre), axis=0)

    # S1a.5  scale floor, used here and by every event threshold.
    finite_sigma = sigma[np.isfinite(sigma) & (sigma > 0)]
    if finite_sigma.size == 0:
        raise ValueError("dark frames do not contain measurable temporal variance")
    scale = np.maximum(
        sigma,
        max(float(np.median(finite_sigma)), float(minimum_sigma_adu)),
    )

    # S1a.6  re-estimate every frame's offset against the master.
    offsets = _grid_offsets(dark_frames_adu, master, stride=offset_stride)
    residual = dark_frames_adu - offsets[:, None, None] - master[None, :, :]

    # S1a.7  clipped mean of the squared residuals; the scale squared where
    # the clip retains at most one frame.
    inlier = np.abs(residual) < variance_clip_sigma * scale[None, :, :]
    retained = np.count_nonzero(inlier, axis=0)
    variance = np.divide(
        np.sum(np.where(inlier, np.square(residual), 0.0), axis=0),
        retained,
        out=np.square(scale),
        where=retained > 1,
    )

    # S1a.8  remove the known inflation from the finite-pilot median.
    variance = variance / (1.0 + np.pi / (2.0 * pilot_index.size))

    # S1a.9  floor at the global median of the finite positive variance.
    positive = variance[np.isfinite(variance) & (variance > 0)]
    if positive.size == 0:
        raise ValueError("dark frames do not contain measurable temporal variance")
    variance = np.maximum(variance, float(np.median(positive)))
    return master, variance, offsets, residual, scale


@dataclass(frozen=True)
class _DarkEvents:
    """Per-pixel event catalogue from the closed-shutter stack.

    Every field except ``size`` is one entry per event pixel; ``size`` is one
    entry per connected component.
    """

    charge_adu: np.ndarray
    zscore: np.ndarray
    row: np.ndarray
    frame: np.ndarray
    at_ceiling: np.ndarray
    size: np.ndarray


def _image_dark_events(
    dark_frames_adu: np.ndarray,
    residual_adu: np.ndarray,
    scale_adu: np.ndarray,
    *,
    saturation_adu: float,
    seed_sigma: float,
    growth_sigma: float,
    ceiling_adu: float,
) -> _DarkEvents:
    # Algorithm S1a steps 1.10-1.11: the thresholds use the floored per-pixel
    # scale of step 1.5, not the square root of the clipped variance.
    zscore = residual_adu / scale_adu[None, :, :]
    zscore[dark_frames_adu >= saturation_adu] = -np.inf
    event_charges: list[float] = []
    event_zscores: list[float] = []
    event_rows: list[int] = []
    event_frames: list[int] = []
    event_at_ceiling: list[bool] = []
    event_sizes: list[int] = []
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
            # Step 6 drops an event whose peak reached the ADC ceiling: its
            # charge is a lower bound, not a measurement.
            at_ceiling = bool(
                np.max(dark_frames_adu[frame][component]) >= ceiling_adu
            )
            event_charges.extend(charges.tolist())
            event_zscores.extend(zscore[frame][component].tolist())
            event_rows.extend(rows.tolist())
            event_frames.extend([frame] * charges.size)
            event_at_ceiling.extend([at_ceiling] * charges.size)
            event_sizes.append(int(charges.size))
    return _DarkEvents(
        charge_adu=np.asarray(event_charges, dtype=np.float64),
        zscore=np.asarray(event_zscores, dtype=np.float64),
        row=np.asarray(event_rows, dtype=np.int32),
        frame=np.asarray(event_frames, dtype=np.int32),
        at_ceiling=np.asarray(event_at_ceiling, dtype=bool),
        size=np.asarray(event_sizes, dtype=np.int32),
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


def _plain_profile_spectrum(
    frame_adu: np.ndarray,
    background_adu: np.ndarray,
    model: ImageModel,
    *,
    gain: float,
    iterations: int = 3,
) -> np.ndarray:
    """No-rejection profile-weighted extraction on the calibrated support."""

    support_y = model.support_y
    support_x = model.support_x
    weights = model.support_profile
    n_columns = int(model.profile.shape[1])
    signal = (np.asarray(frame_adu, dtype=np.float64)[support_y, support_x]
              - np.asarray(background_adu, dtype=np.float64)[support_y, support_x])
    detector = np.asarray(model.support_detector_variance_adu2, dtype=np.float64)
    variance = detector.copy()
    spectrum = np.zeros(n_columns, dtype=np.float64)
    for _ in range(max(1, int(iterations))):
        precision = np.bincount(
            support_x, weights=weights * weights / variance, minlength=n_columns)
        right_hand = np.bincount(
            support_x, weights=weights * signal / variance, minlength=n_columns)
        spectrum = np.divide(
            right_hand, precision,
            out=np.zeros_like(right_hand), where=precision > 0)
        optical_mean = weights * np.clip(spectrum[support_x], 0.0, None)
        variance = detector + optical_mean / gain
    return spectrum


def _profile_residual(
    sample_frames_adu: np.ndarray,
    background_adu: np.ndarray,
    model: ImageModel,
    *,
    gain: float,
    maximum_frames: int,
) -> np.ndarray:
    """Median fractional profile residual, Algorithm S1a steps 17-18.

    The inference model carries no term for the optical profile being
    slightly wrong, so a static shape error in P has nowhere to go but the
    particle-charge channel, where it is read as a cosmic ray.  The residual
    is taken as a fraction of the extracted spectrum rather than in ADU
    because the error scales with the signal: it is a shape error, not an
    offset, so a raw ADU map would not transfer between frames or exposures.
    """

    profile = np.asarray(model.profile, dtype=np.float64)
    support = profile > 0.0
    background = np.asarray(background_adu, dtype=np.float64)
    stack: list[np.ndarray] = []
    for frame in np.asarray(sample_frames_adu[:maximum_frames], dtype=np.float64):
        spectrum = _plain_profile_spectrum(frame, background, model, gain=gain)
        usable = np.isfinite(spectrum) & (spectrum > 0.0)
        if not np.any(usable):
            continue
        residual = np.zeros_like(profile)
        observed = frame - background
        with np.errstate(invalid="ignore", divide="ignore"):
            residual[:, usable] = (
                observed[:, usable] - profile[:, usable] * spectrum[usable]
            ) / spectrum[usable]
        residual[~support] = 0.0
        stack.append(residual)
    if not stack:
        return np.zeros_like(profile)
    return np.median(np.stack(stack), axis=0)


def _corrected_profile(
    profile: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    """P' = normalise(max(0, P + rho)) on the existing support."""

    profile = np.asarray(profile, dtype=np.float64)
    support = profile > 0.0
    updated = np.clip(profile + np.asarray(residual, dtype=np.float64), 0.0, None)
    updated[~support] = 0.0
    total = updated.sum(axis=0)
    positive = total > 0.0
    updated[:, positive] /= total[positive]
    # A column whose correction annihilates the profile keeps the original.
    updated[:, ~positive] = profile[:, ~positive]
    return updated


def calibrate_image(
    dark_frames_adu: np.ndarray,
    sample_profile_frames_adu: np.ndarray,
    instrument_background_adu: np.ndarray,
    *,
    gain_e_per_adu: float = 0.7,
    profile_training_frames: int = 25,
    mad_to_sigma: float = 1.4826,
    saturation_adu: float = 65535.0,
    pilot_frames: int = 31,
    offset_stride: int = 8,
    variance_clip_sigma: float = 6.0,
    minimum_sigma_adu: float = 0.5,
    event_seed_sigma: float = 8.0,
    event_grow_sigma: float = 3.0,
    trace_half_width_rows: int = 12,
    profile_smoothing_width_columns: int = 9,
    ridge_smoothing_width_columns: int = 15,
    row_smoothing_width: int = 9,
    maximum_amplitude_components: int = 5,
    charge_ceiling_adu: float = 65000.0,
    inner_validation_fraction: float = 0.20,
    charge_prior_seed: int = 0,
) -> ImageCalibration:
    """Build a 2-D model from closed-shutter and sample training stacks."""

    dark = _validated_image_stack("dark_frames_adu", dark_frames_adu)
    sample = _validated_image_stack(
        "sample_profile_frames_adu",
        sample_profile_frames_adu,
    )
    if sample.shape[1:] != dark.shape[1:]:
        raise ValueError("dark and sample-profile images must match")
    # Algorithm S1a step 8: the instrumental background is a required input.
    # This calibration performs no exposure-series fit, so it must fail when
    # no map is supplied rather than silently substituting something else.
    instrument_background = np.asarray(instrument_background_adu, dtype=np.float64)
    if instrument_background.shape != dark.shape[1:]:
        raise ValueError(
            "instrument_background_adu must be a 2-D map matching the detector"
        )
    if np.any(~np.isfinite(instrument_background)):
        raise ValueError("instrument_background_adu must contain finite values")
    gain = _positive_float("gain_e_per_adu", gain_e_per_adu)
    mad_scale = _positive_float("mad_to_sigma", mad_to_sigma)
    saturation = _positive_float("saturation_adu", saturation_adu)
    n_pilot = _positive_integer("pilot_frames", pilot_frames)
    stride = _positive_integer("offset_stride", offset_stride)
    clip_sigma = _positive_float("variance_clip_sigma", variance_clip_sigma)
    sigma_floor = _positive_float("minimum_sigma_adu", minimum_sigma_adu)
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
    maximum_components = _positive_integer(
        "maximum_amplitude_components",
        maximum_amplitude_components,
    )
    training_frames = _positive_integer(
        "profile_training_frames",
        profile_training_frames,
    )
    ceiling = _positive_float("charge_ceiling_adu", charge_ceiling_adu)
    inner_fraction = float(inner_validation_fraction)
    if not 0.0 < inner_fraction < 1.0:
        raise ValueError("inner_validation_fraction must lie strictly in (0, 1)")
    prior_seed = int(charge_prior_seed)

    (
        master,
        dark_variance,
        dark_offsets,
        dark_residual,
        dark_scale,
    ) = _image_dark_statistics(
        dark,
        mad_to_sigma=mad_scale,
        pilot_frames=n_pilot,
        offset_stride=stride,
        variance_clip_sigma=clip_sigma,
        minimum_sigma_adu=sigma_floor,
    )
    events = _image_dark_events(
        dark,
        dark_residual,
        dark_scale,
        saturation_adu=saturation,
        seed_sigma=seed_threshold,
        growth_sigma=growth_threshold,
        ceiling_adu=ceiling,
    )
    if events.size.size == 0:
        raise ValueError("closed-shutter stack contains no detected particle events")

    # Algorithm S1a steps 5-6.  The amplitude sample drops ceiling events and
    # non-positive charges; each surviving pixel carries the censoring
    # threshold L_j = 3 Chat_j / nu_j, and the component count is chosen on a
    # held-out partition of the dark frames rather than fixed in advance.
    usable = (~events.at_ceiling) & (events.charge_adu > 0.0)
    charges = events.charge_adu[usable]
    if charges.size < 2 * maximum_components:
        raise ValueError("too few event pixels to fit the charge prior")
    thresholds = np.minimum(
        3.0 * charges / np.clip(events.zscore[usable], 1e-12, None),
        charges,
    )
    validation_start = int(
        np.ceil((1.0 - inner_fraction) * float(dark.shape[0]))
    )
    inner_fit = events.frame[usable] < max(1, validation_start)
    if not inner_fit.any() or inner_fit.all():
        inner_fit = np.arange(charges.size) % 5 != 4
    prior_fit = select_charge_prior(
        charges,
        thresholds,
        inner_fit,
        maximum_components=maximum_components,
        random_seed=prior_seed,
    )
    amplitude_scales = np.asarray(prior_fit["scales_adu"], dtype=np.float64)
    amplitude_weights = np.asarray(prior_fit["weights"], dtype=np.float64)
    event_sizes = events.size
    event_rows = events.row
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

    # D^sub = D + B^inst, Algorithm S1a step 8. Everything downstream of the
    # template -- b_x, the profile candidates and the residual correction --
    # is measured against this, not against the master dark alone.
    subtracted_dark = master + instrument_background
    sample_template = _aligned_image_template(sample, subtracted_dark)
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
    # Algorithm S1a step 16: the shot term comes from the off-trace diffuse
    # photo level, not from the dark master.  The master is overwhelmingly
    # the detector pedestal, which carries no shot noise; using it inflates
    # the detector variance by roughly the pedestal-to-diffuse ratio and
    # collapses every event z-score in Algorithm S1b.
    detector_variance = dark_variance + np.clip(diffuse, 0.0, None)[None, :] / gain
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
        # Algorithm S1a step 19: discard rows below 0.002 of their column
        # maximum.  build_image_model already defaults to this; passing 0.0
        # kept every trace-mask row, each contributing a near-zero profile
        # weight with a full variance term.
        relative_profile_floor=0.002,
    )

    # Algorithm S1a steps 17-18: correct the profile against its own
    # training frames before shipping it.  The first model is built only to
    # supply the support and the extraction weights the residual is measured
    # with; what the caller receives is the corrected profile.
    # The subtracted background is D + B^inst. b_x is a separate calibrated
    # quantity: it sets the per-column baseline of the profile candidates and
    # the diffuse shot term, and is deliberately not subtracted here, since
    # doing so would remove real diffuse signal at correction time.
    background = subtracted_dark
    profile_residual = _profile_residual(
        sample,
        background,
        model,
        gain=gain,
        maximum_frames=training_frames,
    )
    model = build_image_model(
        optical_template,
        trace_mask,
        particle_prior,
        detector_variance_adu2=detector_variance,
        spatial_profile_template=_corrected_profile(model.profile, profile_residual),
        template_spectrum_adu=template_spectrum,
        relative_profile_floor=0.002,
    )
    return ImageCalibration(
        background_adu=background,
        model=model,
        dark_master_adu=master,
        dark_variance_adu2=dark_variance,
        dark_frame_offsets_adu=dark_offsets,
        diffuse_photo_adu=diffuse,
        n_dark_events=int(event_sizes.size),
        n_dark_event_pixels=int(events.charge_adu.size),
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

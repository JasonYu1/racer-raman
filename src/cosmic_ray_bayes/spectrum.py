"""Cosmic-ray correction for row-summed spectra.

The cleaner combines two estimators.  An empirical-Bayes branch subtracts
particle charge away from bright spectral bands.  On wide, bright bands an
iteratively masked smooth baseline is more reliable, so a deterministic regime
switch selects that branch column by column.

Only NumPy arrays are needed at inference.  Calibration is built separately
from positive per-column particle charges, event widths, and an independently
measured event rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.signal import find_peaks, peak_widths
from scipy.sparse.linalg import spsolve
from scipy.special import expit, logsumexp

__all__ = [
    "BayesianSpectrumConfig",
    "HybridSpectrumConfig",
    "MaskedBaselineConfig",
    "SpectrumCalibration",
    "build_spectrum_calibration",
    "clean_spectrum",
]


@dataclass(frozen=True)
class BayesianSpectrumConfig:
    """Settings for empirical-Bayes charge subtraction."""

    continuum_penalty: float = 0.0
    continuum_penalty_order: int = 2
    continuum_mask_guard: int = 1
    broad_smoothing_sigma: float = 2.0
    broad_robust_sigma: float = 3.0
    broad_robust_iterations: int = 2
    robust_continuum_first_pass_only: bool = False
    maximum_iterations: int = 8
    convergence_adu: float = 1.0e-3
    adjacency_radius: int = 1
    adjacency_incidence: float = 0.5
    confirmation_threshold: float = 0.5
    report_threshold: float = 0.3
    subtraction_threshold: float = 0.0
    local_blend: bool = False
    local_estimate_sigma: float = 4.0
    evaluation_residual_sigma: float = 0.0
    width_run_sigma: float = 1.0
    width_baseline_columns: int = 101
    width_assignment_rel_height: float = 0.95
    # The supplement describes no width prior on the posterior odds, and the
    # released hybrid configuration sets this to zero. The class default now
    # matches it, so nothing the published method does not describe is active
    # unless a caller asks for it explicitly.
    width_penalty_weight: float = 0.0
    maximum_particle_width: float = 13.0
    limit_accepted_run: bool = False
    continuum_maximum_particle_width: int = 10

    def __post_init__(self) -> None:
        nonnegative = (
            "continuum_penalty",
            "evaluation_residual_sigma",
            "subtraction_threshold",
            "width_penalty_weight",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

        positive = (
            "broad_smoothing_sigma",
            "broad_robust_sigma",
            "convergence_adu",
            "local_estimate_sigma",
            "width_run_sigma",
            "maximum_particle_width",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        for name in (
            "continuum_penalty_order",
            "maximum_iterations",
            "width_baseline_columns",
            "continuum_maximum_particle_width",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "continuum_mask_guard",
            "broad_robust_iterations",
            "adjacency_radius",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be nonnegative")

        if not 0.0 < float(self.adjacency_incidence) < 1.0:
            raise ValueError("adjacency_incidence must lie in (0, 1)")
        for name in (
            "confirmation_threshold",
            "report_threshold",
            "subtraction_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not 0.0 < float(self.width_assignment_rel_height) <= 1.0:
            raise ValueError("width_assignment_rel_height must lie in (0, 1]")


@dataclass(frozen=True)
class MaskedBaselineConfig:
    """Settings for the iteratively masked smooth-baseline branch."""

    penalty: float = 0.01
    threshold_sigma: float = 5.0
    penalty_order: int = 2
    mask_guard: int = 1
    maximum_iterations: int = 8

    def __post_init__(self) -> None:
        if not np.isfinite(self.penalty) or self.penalty <= 0.0:
            raise ValueError("penalty must be finite and positive")
        if not np.isfinite(self.threshold_sigma) or self.threshold_sigma <= 0.0:
            raise ValueError("threshold_sigma must be finite and positive")
        if int(self.penalty_order) < 1:
            raise ValueError("penalty_order must be positive")
        if int(self.mask_guard) < 0:
            raise ValueError("mask_guard must be nonnegative")
        if int(self.maximum_iterations) < 1:
            raise ValueError("maximum_iterations must be positive")


def _default_hybrid_bayes() -> BayesianSpectrumConfig:
    return BayesianSpectrumConfig(
        evaluation_residual_sigma=2.0,
        subtraction_threshold=0.5,
        maximum_particle_width=13.0,
        local_blend=True,
        limit_accepted_run=True,
        local_estimate_sigma=3.0,
        continuum_penalty=0.3,
        continuum_penalty_order=3,
        width_penalty_weight=0.0,
    )


def _default_hybrid_masked() -> MaskedBaselineConfig:
    return MaskedBaselineConfig(penalty=0.1, mask_guard=0)


@dataclass(frozen=True)
class HybridSpectrumConfig:
    """Settings for the bright-band switch and its two correction branches."""

    use_masked_baseline_on_bands: bool = True
    band_minimum_width: float = 13.0
    band_minimum_sigma: float = 20.0
    band_median_window: int = 21
    bayes: BayesianSpectrumConfig = field(default_factory=_default_hybrid_bayes)
    masked: MaskedBaselineConfig = field(default_factory=_default_hybrid_masked)

    def __post_init__(self) -> None:
        if not isinstance(self.bayes, BayesianSpectrumConfig):
            raise TypeError("bayes must be a BayesianSpectrumConfig")
        if not isinstance(self.masked, MaskedBaselineConfig):
            raise TypeError("masked must be a MaskedBaselineConfig")
        if not np.isfinite(self.band_minimum_width) or self.band_minimum_width <= 0:
            raise ValueError("band_minimum_width must be finite and positive")
        if not np.isfinite(self.band_minimum_sigma) or self.band_minimum_sigma <= 0:
            raise ValueError("band_minimum_sigma must be finite and positive")
        if int(self.band_median_window) < 3:
            raise ValueError("band_median_window must be at least three")


@dataclass(frozen=True)
class SpectrumCalibration:
    """Empirical particle prior for one row-summed exposure.

    ``charge_samples_adu`` contains positive deposited charge per affected
    detector column.  ``width_log_pmf`` is indexed by event width in columns
    and is normalized relative to its most likely width.
    """

    charge_samples_adu: np.ndarray
    column_incidence: float
    width_log_pmf: np.ndarray
    exposure_ms: float
    n_source_events: int

    def __post_init__(self) -> None:
        samples = np.array(self.charge_samples_adu, dtype=float, copy=True)
        width_log_pmf = np.array(self.width_log_pmf, dtype=float, copy=True)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("charge_samples_adu must be a nonempty 1-D array")
        if not np.all(np.isfinite(samples)) or np.any(samples <= 0.0):
            raise ValueError("charge_samples_adu must be finite and positive")
        if width_log_pmf.ndim != 1 or width_log_pmf.size < 2:
            raise ValueError("width_log_pmf must be a 1-D array indexed by width")
        if not np.all(np.isfinite(width_log_pmf)):
            raise ValueError("width_log_pmf must be finite")
        if not 0.0 < float(self.column_incidence) < 1.0:
            raise ValueError("column_incidence must lie in (0, 1)")
        if not np.isfinite(self.exposure_ms) or float(self.exposure_ms) <= 0.0:
            raise ValueError("exposure_ms must be finite and positive")
        if int(self.n_source_events) < 1:
            raise ValueError("n_source_events must be positive")

        samples.setflags(write=False)
        width_log_pmf.setflags(write=False)
        object.__setattr__(self, "charge_samples_adu", samples)
        object.__setattr__(self, "width_log_pmf", width_log_pmf)
        object.__setattr__(self, "column_incidence", float(self.column_incidence))
        object.__setattr__(self, "exposure_ms", float(self.exposure_ms))
        object.__setattr__(self, "n_source_events", int(self.n_source_events))


def build_spectrum_calibration(
    charge_samples_adu: np.ndarray,
    event_widths: np.ndarray,
    *,
    events_per_frame: float,
    detector_columns: int,
    exposure_ms: float,
    maximum_samples: int = 4000,
    seed: int = 20260819,
) -> SpectrumCalibration:
    """Build an array-only row-summed particle calibration.

    The incidence is the event rate multiplied by the observed mean number of
    affected columns per event and divided by the detector width.  Event-width
    probabilities use a half-count floor so an unobserved width remains
    unlikely rather than impossible.
    """

    samples = np.asarray(charge_samples_adu, dtype=float)
    widths_raw = np.asarray(event_widths)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("charge_samples_adu must be a nonempty 1-D array")
    if not np.all(np.isfinite(samples)) or np.any(samples <= 0.0):
        raise ValueError("charge_samples_adu must be finite and positive")
    if widths_raw.ndim != 1 or widths_raw.size == 0:
        raise ValueError("event_widths must be a nonempty 1-D array")
    if not np.issubdtype(widths_raw.dtype, np.number):
        raise ValueError("event_widths must be numeric")
    widths_float = np.asarray(widths_raw, dtype=float)
    if not np.all(np.isfinite(widths_float)) or np.any(widths_float < 1.0):
        raise ValueError("event_widths must be finite and at least one")
    if np.any(widths_float != np.floor(widths_float)):
        raise ValueError("event_widths must contain whole column counts")
    if int(detector_columns) != detector_columns or int(detector_columns) < 1:
        raise ValueError("detector_columns must be a positive integer")
    detector_columns = int(detector_columns)
    if np.any(widths_float > detector_columns):
        raise ValueError("event widths cannot exceed detector_columns")
    widths = widths_float.astype(int)
    if int(maximum_samples) < 1:
        raise ValueError("maximum_samples must be positive")
    if not np.isfinite(events_per_frame) or float(events_per_frame) < 0.0:
        raise ValueError("events_per_frame must be finite and nonnegative")
    if not np.isfinite(exposure_ms) or float(exposure_ms) <= 0.0:
        raise ValueError("exposure_ms must be finite and positive")

    observed_column_samples = int(samples.size)
    retained = samples.copy()
    if retained.size > int(maximum_samples):
        rng = np.random.default_rng(seed)
        retained = np.sort(rng.choice(retained, int(maximum_samples), replace=False))

    maximum_width = int(np.max(widths))
    counts = np.bincount(widths, minlength=maximum_width + 1).astype(float)
    counts[0] = 0.0
    counts += 0.5
    width_log_pmf = np.log(counts) - np.log(counts.sum())
    width_log_pmf -= np.max(width_log_pmf)

    columns_per_event = observed_column_samples / float(widths.size)
    incidence = float(events_per_frame) * columns_per_event / detector_columns
    incidence = float(np.clip(incidence, 1.0e-12, 1.0 - 1.0e-12))
    return SpectrumCalibration(
        charge_samples_adu=retained,
        column_incidence=incidence,
        width_log_pmf=width_log_pmf,
        exposure_ms=float(exposure_ms),
        n_source_events=int(widths.size),
    )


def _validated_measurement(
    spectrum: np.ndarray,
    variance_adu2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(spectrum, dtype=float)
    variance = np.asarray(variance_adu2, dtype=float)
    if values.ndim != 1 or values.size < 5:
        raise ValueError("spectrum must be a 1-D array of length at least five")
    if variance.shape != values.shape:
        raise ValueError("variance_adu2 must match the spectrum")
    if not np.all(np.isfinite(values)):
        raise ValueError("spectrum must be finite")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("variance_adu2 must be finite and positive")
    return values, variance


def spectral_peak_width(
    spectrum: np.ndarray,
    variance_adu2: np.ndarray,
    threshold_sigma: float,
    baseline_columns: int,
    assignment_relative_height: float = 0.95,
) -> np.ndarray:
    """Return the half-prominence width assigned to each peak column."""

    values, variance = _validated_measurement(spectrum, variance_adu2)
    if not np.isfinite(threshold_sigma) or float(threshold_sigma) <= 0.0:
        raise ValueError("threshold_sigma must be finite and positive")
    if int(baseline_columns) < 1:
        raise ValueError("baseline_columns must be positive")
    if not 0.0 < float(assignment_relative_height) <= 1.0:
        raise ValueError("assignment_relative_height must lie in (0, 1]")

    size = max(int(baseline_columns) | 1, 3)
    baseline = ndi.median_filter(values, size=size, mode="nearest")
    feature = values - baseline
    minimum_height = float(threshold_sigma) * np.sqrt(np.median(variance))
    result = np.zeros(values.size, dtype=float)
    peaks, _ = find_peaks(feature, height=minimum_height)
    if peaks.size == 0:
        return result

    widths = peak_widths(feature, peaks, rel_height=0.5)[0]
    _, _, left_edges, right_edges = peak_widths(
        feature,
        peaks,
        rel_height=float(assignment_relative_height),
    )
    for width, left, right in zip(
        widths,
        left_edges,
        right_edges,
        strict=False,
    ):
        start = max(0, int(np.floor(left)))
        stop = min(values.size, int(np.ceil(right)) + 1)
        result[start:stop] = np.maximum(result[start:stop], float(width))
    return result


def _width_log_factor(
    widths: np.ndarray,
    width_log_pmf: np.ndarray,
    weight: float,
) -> np.ndarray:
    indices = np.clip(
        np.rint(widths).astype(int),
        0,
        len(width_log_pmf) - 1,
    )
    factor = np.where(widths > 0.0, width_log_pmf[indices], 0.0)
    return float(weight) * np.minimum(factor, 0.0)


def _trim_long_runs(estimate: np.ndarray, limit: int) -> np.ndarray:
    """Keep the highest-charge window within each overlong accepted run."""

    trimmed = np.asarray(estimate, dtype=float).copy()
    accepted = trimmed > 0.0
    if not np.any(accepted) or limit < 1:
        return trimmed
    padded = np.r_[False, accepted, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    for start, stop in zip(starts, stops, strict=False):
        length = int(stop - start)
        if length <= limit:
            continue
        block = trimmed[start:stop]
        window_charge = np.convolve(block, np.ones(limit), mode="valid")
        offset = int(np.argmax(window_charge))
        keep = np.zeros(length, dtype=bool)
        keep[offset : offset + limit] = True
        block[~keep] = 0.0
        trimmed[start:stop] = block
    return trimmed


def _difference_operator(size: int, order: int) -> sparse.csc_matrix:
    difference = sparse.eye(size, format="csc")
    for _ in range(int(order)):
        difference = difference[1:] - difference[:-1]
    return difference


def _penalized_continuum(
    values: np.ndarray,
    variance: np.ndarray,
    claimed: np.ndarray,
    penalty: float,
    order: int,
    guard: int,
) -> np.ndarray:
    """Fit a smooth continuum while holding claimed event columns out."""

    mask = np.asarray(claimed, dtype=bool)
    if guard > 0 and np.any(mask):
        mask = ndi.binary_dilation(mask, np.ones(2 * int(guard) + 1, dtype=bool))
    weight = np.where(mask, 0.0, 1.0 / variance)
    if not np.any(weight > 0.0):
        return values.copy()
    weight /= np.max(weight)
    difference = _difference_operator(values.size, order)
    left = sparse.diags(weight) + float(penalty) * (difference.T @ difference)
    return np.asarray(spsolve(left.tocsc(), weight * values), dtype=float)


def _robust_continuum(
    values: np.ndarray,
    variance: np.ndarray,
    config: BayesianSpectrumConfig,
    *,
    robust: bool = True,
) -> np.ndarray:
    """Estimate broad spectral structure without following narrow spikes."""

    base_weight = np.clip(np.median(variance) / variance, 0.05, 20.0)
    sigma = float(config.broad_smoothing_sigma)
    if not robust:
        numerator = ndi.gaussian_filter1d(
            base_weight * values,
            sigma=sigma,
            mode="nearest",
        )
        denominator = ndi.gaussian_filter1d(
            base_weight,
            sigma=sigma,
            mode="nearest",
        )
        return numerator / np.maximum(denominator, np.finfo(float).eps)

    broad_width = spectral_peak_width(
        values,
        variance,
        config.broad_robust_sigma,
        config.width_baseline_columns,
        config.width_assignment_rel_height,
    )
    robust_weight = np.ones_like(values)
    prediction = ndi.gaussian_filter1d(values, sigma=sigma, mode="nearest")
    for _ in range(int(config.broad_robust_iterations)):
        weight = base_weight * robust_weight
        numerator = ndi.gaussian_filter1d(
            weight * values,
            sigma=sigma,
            mode="nearest",
        )
        denominator = ndi.gaussian_filter1d(
            weight,
            sigma=sigma,
            mode="nearest",
        )
        prediction = numerator / np.maximum(denominator, np.finfo(float).eps)
        positive_z = np.clip(
            (values - prediction) / np.sqrt(variance),
            0.0,
            None,
        )
        robust_weight = np.minimum(
            1.0,
            (
                float(config.broad_robust_sigma)
                / np.maximum(positive_z, np.finfo(float).eps)
            )
            ** 2,
        )
        robust_weight = np.where(
            broad_width > float(config.continuum_maximum_particle_width),
            1.0,
            robust_weight,
        )
    return prediction


def _posterior_and_charge(
    residual: np.ndarray,
    variance: np.ndarray,
    charge_samples: np.ndarray,
    prior_log_odds: np.ndarray,
    evaluate: np.ndarray,
    block_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return event probability and conditional charge moments."""

    probability = np.zeros(residual.size, dtype=float)
    mean_charge = np.zeros(residual.size, dtype=float)
    charge_variance = np.zeros(residual.size, dtype=float)
    indices = np.flatnonzero(evaluate)
    if indices.size == 0:
        return probability, mean_charge, charge_variance

    log_count = np.log(charge_samples.size)
    charge_squared = charge_samples**2
    for start in range(0, indices.size, int(block_size)):
        selected = indices[start : start + int(block_size)]
        current_residual = residual[selected, None]
        current_variance = variance[selected, None]
        log_weight = (
            2.0 * current_residual * charge_samples[None, :] - charge_squared[None, :]
        ) / (2.0 * current_variance)
        normalizer = logsumexp(log_weight, axis=1)
        probability[selected] = expit(prior_log_odds[selected] + normalizer - log_count)
        weights = np.exp(log_weight - normalizer[:, None])
        first_moment = weights @ charge_samples
        second_moment = weights @ charge_squared
        mean_charge[selected] = first_moment
        charge_variance[selected] = np.maximum(
            second_moment - first_moment**2,
            0.0,
        )
    return probability, mean_charge, charge_variance


def _bayesian_despike(
    values: np.ndarray,
    variance: np.ndarray,
    calibration: SpectrumCalibration,
    config: BayesianSpectrumConfig,
) -> dict[str, Any]:
    """Subtract posterior mean particle charge from one spectrum."""

    charge_samples = calibration.charge_samples_adu
    marginal = float(calibration.column_incidence)
    marginal_log_odds = np.log(marginal) - np.log1p(-marginal)
    adjacent = float(config.adjacency_incidence)
    adjacent_log_odds = np.log(adjacent) - np.log1p(-adjacent)

    estimate = np.zeros_like(values)
    probability = np.zeros_like(values)
    continuum = values.copy()
    mean_charge = np.zeros_like(values)
    charge_variance = np.zeros_like(values)
    widths = np.zeros_like(values)
    width_factor = np.zeros_like(values)
    iterations = 0

    for iteration in range(1, int(config.maximum_iterations) + 1):
        iterations = iteration
        if float(config.continuum_penalty) > 0.0:
            claimed = estimate > 0.0
            if np.any(claimed):
                continuum = _penalized_continuum(
                    values,
                    variance,
                    claimed,
                    float(config.continuum_penalty),
                    int(config.continuum_penalty_order),
                    int(config.continuum_mask_guard),
                )
            else:
                continuum = _robust_continuum(values, variance, config)
        else:
            use_robust = not (
                config.robust_continuum_first_pass_only and np.any(estimate > 0.0)
            )
            continuum = _robust_continuum(
                values - estimate,
                variance,
                config,
                robust=use_robust,
            )
        residual = values - continuum

        widths = spectral_peak_width(
            values,
            variance,
            config.width_run_sigma,
            config.width_baseline_columns,
            config.width_assignment_rel_height,
        )
        width_factor = _width_log_factor(
            widths,
            calibration.width_log_pmf,
            config.width_penalty_weight,
        )

        prior_log_odds = np.full(values.size, marginal_log_odds, dtype=float)
        confirmed = probability >= float(config.confirmation_threshold)
        if config.adjacency_radius > 0 and np.any(confirmed):
            grown = ndi.binary_dilation(
                confirmed,
                np.ones(2 * int(config.adjacency_radius) + 1, dtype=bool),
            )
            neighbours = grown & ~confirmed & (residual > 0.0)
            prior_log_odds[neighbours] = adjacent_log_odds

        evaluate = residual > (
            float(config.evaluation_residual_sigma) * np.sqrt(variance)
        )
        probability, mean_charge, charge_variance = _posterior_and_charge(
            residual,
            variance,
            charge_samples,
            prior_log_odds + width_factor,
            evaluate,
        )
        probability = np.where(
            widths > float(config.maximum_particle_width),
            0.0,
            probability,
        )
        updated = np.clip(probability * mean_charge, 0.0, None)
        if float(config.subtraction_threshold) > 0.0:
            updated = np.where(
                probability >= float(config.subtraction_threshold),
                updated,
                0.0,
            )
        if config.limit_accepted_run:
            updated = _trim_long_runs(
                updated,
                int(round(float(config.maximum_particle_width))),
            )

        shift = float(np.max(np.abs(updated - estimate)))
        estimate = updated
        if shift <= float(config.convergence_adu):
            break

    corrected = values - estimate
    blend_weight = np.zeros_like(values)
    if config.local_blend:
        removed_variance = (
            probability * charge_variance
            + probability * (1.0 - probability) * mean_charge**2
        )
        observation_variance = variance + removed_variance
        good = (estimate <= 0.0).astype(float)
        local_sigma = max(float(config.local_estimate_sigma), 0.5)
        numerator = ndi.gaussian_filter1d(
            good * values,
            sigma=local_sigma,
            mode="nearest",
        )
        denominator = ndi.gaussian_filter1d(
            good,
            sigma=local_sigma,
            mode="nearest",
        )
        local_estimate = numerator / np.maximum(denominator, np.finfo(float).eps)
        sum_square = ndi.gaussian_filter1d(
            good,
            sigma=local_sigma / np.sqrt(2.0),
            mode="nearest",
        ) / (2.0 * local_sigma * np.sqrt(np.pi))
        effective_samples = np.maximum(
            denominator**2 / np.maximum(sum_square, np.finfo(float).eps),
            1.0,
        )
        local_variance = np.maximum(
            variance / effective_samples,
            np.finfo(float).eps,
        )
        usable = denominator > 1.0e-3
        blend_weight = np.where(
            usable,
            (1.0 / observation_variance)
            / (1.0 / observation_variance + 1.0 / local_variance),
            1.0,
        )
        blended = blend_weight * corrected + (1.0 - blend_weight) * np.where(
            usable,
            local_estimate,
            corrected,
        )
        corrected = np.where(estimate > 0.0, blended, values)

    return {
        "spectrum": corrected,
        "mask": probability >= float(config.report_threshold),
        "posterior": probability,
        "event_model_adu": estimate,
        "continuum_adu": continuum,
        "width_log_factor": width_factor,
        "excursion_width": widths,
        "charge_variance_adu2": charge_variance,
        "local_blend_weight": blend_weight,
        "iterations": iterations,
        "converged": bool(iterations < int(config.maximum_iterations)),
        "config": config,
    }


def _penalized_fit(
    values: np.ndarray,
    weight: np.ndarray,
    penalty: float,
    order: int,
) -> np.ndarray:
    difference = _difference_operator(values.size, order)
    left = sparse.diags(weight) + float(penalty) * (difference.T @ difference)
    return np.asarray(spsolve(left.tocsc(), weight * values), dtype=float)


def _masked_baseline_despike(
    values: np.ndarray,
    variance: np.ndarray,
    config: MaskedBaselineConfig,
) -> dict[str, Any]:
    """Replace positive outliers with a baseline fitted around their mask."""

    deviation = np.sqrt(variance)
    reciprocal = 1.0 / variance
    reciprocal /= np.max(reciprocal)
    flagged = np.zeros(values.size, dtype=bool)
    baseline = values.copy()
    iterations = 0

    for iteration in range(1, int(config.maximum_iterations) + 1):
        iterations = iteration
        held_out = flagged
        if config.mask_guard > 0 and np.any(flagged):
            held_out = ndi.binary_dilation(
                flagged,
                np.ones(2 * int(config.mask_guard) + 1, dtype=bool),
            )
        weight = np.where(held_out, 0.0, reciprocal)
        if not np.any(weight > 0.0):
            break
        baseline = _penalized_fit(
            values,
            weight,
            config.penalty,
            config.penalty_order,
        )
        updated = (values - baseline) > float(config.threshold_sigma) * deviation
        if np.array_equal(updated, flagged):
            break
        flagged = updated

    corrected = np.where(flagged, baseline, values)
    return {
        "spectrum": corrected,
        "mask": flagged,
        "baseline_adu": baseline,
        "event_model_adu": np.where(flagged, values - baseline, 0.0),
        "iterations": iterations,
        "converged": bool(iterations < int(config.maximum_iterations)),
        "config": config,
    }


def _bright_band_mask(
    values: np.ndarray,
    variance: np.ndarray,
    config: HybridSpectrumConfig,
) -> np.ndarray:
    window = max(int(config.band_median_window) | 1, 3)
    reference = ndi.median_filter(values, size=window, mode="nearest")
    widths = spectral_peak_width(
        reference,
        variance,
        float(config.band_minimum_sigma),
        int(config.bayes.width_baseline_columns),
        float(config.bayes.width_assignment_rel_height),
    )
    return widths > float(config.band_minimum_width)


def clean_spectrum(
    spectrum: np.ndarray,
    variance_adu2: np.ndarray,
    calibration: SpectrumCalibration,
    *,
    config: HybridSpectrumConfig | None = None,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Remove cosmic-ray events from one dark- and offset-corrected spectrum.

    The returned array preserves every column that neither branch corrects
    bit-for-bit.  Set ``return_details=True`` to also receive masks, branch
    assignments, posterior values, and the two underlying branch results.
    """

    values, variance = _validated_measurement(spectrum, variance_adu2)
    if not isinstance(calibration, SpectrumCalibration):
        raise TypeError("calibration must be a SpectrumCalibration")
    if config is None:
        config = HybridSpectrumConfig()
    elif not isinstance(config, HybridSpectrumConfig):
        raise TypeError("config must be a HybridSpectrumConfig")

    bayesian = _bayesian_despike(values, variance, calibration, config.bayes)
    if not config.use_masked_baseline_on_bands:
        details = dict(bayesian)
        details["band_mask"] = np.zeros(values.size, dtype=bool)
        details["branch"] = np.zeros(values.size, dtype=np.uint8)
        details["config"] = config
    else:
        masked = _masked_baseline_despike(values, variance, config.masked)
        band_mask = _bright_band_mask(values, variance, config)
        corrected = np.where(
            band_mask,
            np.asarray(masked["spectrum"], dtype=float),
            np.asarray(bayesian["spectrum"], dtype=float),
        )
        flagged = np.where(
            band_mask,
            np.asarray(masked["mask"], dtype=bool),
            np.asarray(bayesian["mask"], dtype=bool),
        )
        details = {
            "spectrum": corrected,
            "mask": flagged,
            "event_model_adu": values - corrected,
            "band_mask": band_mask,
            "branch": band_mask.astype(np.uint8),
            "bayes_result": bayesian,
            "masked_result": masked,
            "config": config,
        }

    cleaned = np.asarray(details["spectrum"], dtype=float)
    if return_details:
        return cleaned, details
    return cleaned

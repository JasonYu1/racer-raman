"""Profile-aware empirical-Bayes correction for 2-D detector images."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np
from scipy import ndimage as ndi
from scipy.special import gammaln, log_ndtr, logsumexp, xlogy

__all__ = [
    "ParticlePrior",
    "ImageModel",
    "ImageConfig",
    "build_image_model",
    "clean_image",
]

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class ParticlePrior:
    """Particle incidence, charge mixture, and morphology from dark frames."""

    row_probability: np.ndarray
    amplitude_scales_adu: np.ndarray
    amplitude_weights: np.ndarray
    maximum_event_pixels: int | None = None

    def __post_init__(self) -> None:
        row_probability = np.array(self.row_probability, dtype=np.float64, copy=True)
        scales = np.array(self.amplitude_scales_adu, dtype=np.float64, copy=True)
        weights = np.array(self.amplitude_weights, dtype=np.float64, copy=True)
        if row_probability.ndim != 1 or row_probability.size == 0:
            raise ValueError("row_probability must be a nonempty 1-D array")
        if np.any(~np.isfinite(row_probability)) or np.any(
            (row_probability <= 0) | (row_probability >= 1)
        ):
            raise ValueError(
                "row probabilities must be finite and lie strictly between zero and one"
            )
        if (
            scales.ndim != 1
            or scales.size == 0
            or np.any(~np.isfinite(scales))
            or np.any(scales <= 0)
        ):
            raise ValueError(
                "amplitude scales must be a nonempty finite positive 1-D array"
            )
        if (
            weights.shape != scales.shape
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError(
                "amplitude weights must be finite, nonnegative, and match scales"
            )
        maximum_pixels = self.maximum_event_pixels
        if maximum_pixels is not None:
            if isinstance(maximum_pixels, bool | np.bool_) or not isinstance(
                maximum_pixels, int | np.integer
            ):
                raise ValueError(
                    "maximum_event_pixels must be a positive integer or None"
                )
            maximum_pixels = int(maximum_pixels)
            if maximum_pixels <= 0:
                raise ValueError(
                    "maximum_event_pixels must be a positive integer or None"
                )
        row_probability.setflags(write=False)
        scales.setflags(write=False)
        normalized_weights = weights / weights.sum()
        normalized_weights.setflags(write=False)
        object.__setattr__(self, "row_probability", row_probability)
        object.__setattr__(self, "amplitude_scales_adu", scales)
        object.__setattr__(self, "amplitude_weights", normalized_weights)
        object.__setattr__(self, "maximum_event_pixels", maximum_pixels)


@dataclass(frozen=True)
class ImageModel:
    """Calibrated profile, detector variance, and particle prior."""

    profile: np.ndarray
    template_spectrum: np.ndarray
    valid_columns: np.ndarray
    support_y: np.ndarray
    support_x: np.ndarray
    support_profile: np.ndarray
    support_event_probability: np.ndarray
    particle_prior: ParticlePrior
    support_detector_variance_adu2: np.ndarray
    spectrum_prior_fraction: float
    spectrum_prior_floor_adu: float

    def __post_init__(self) -> None:
        profile = np.array(self.profile, copy=True)
        template_spectrum = np.array(
            self.template_spectrum, dtype=np.float64, copy=True
        )
        valid_columns = np.array(self.valid_columns, dtype=bool, copy=True)
        support_y = np.array(self.support_y, dtype=np.int32, copy=True)
        support_x = np.array(self.support_x, dtype=np.int32, copy=True)
        support_profile = np.array(self.support_profile, dtype=np.float64, copy=True)
        support_event_probability = np.array(
            self.support_event_probability, dtype=np.float64, copy=True
        )
        support_variance = np.array(
            self.support_detector_variance_adu2, dtype=np.float64, copy=True
        )

        if profile.ndim != 2 or 0 in profile.shape:
            raise ValueError("profile must be a nonempty 2-D array")
        if np.any(~np.isfinite(profile)) or np.any(profile < 0):
            raise ValueError("profile must be finite and nonnegative")
        rows, columns = profile.shape
        if template_spectrum.shape != (columns,):
            raise ValueError("template_spectrum must have one value per column")
        if np.any(~np.isfinite(template_spectrum)) or np.any(template_spectrum < 0):
            raise ValueError("template_spectrum must be finite and nonnegative")
        if valid_columns.shape != (columns,):
            raise ValueError("valid_columns must have one value per column")
        if not isinstance(self.particle_prior, ParticlePrior):
            raise TypeError("particle_prior must be a ParticlePrior")
        if self.particle_prior.row_probability.size != rows:
            raise ValueError("particle row prior does not match profile rows")

        support_size = support_y.size
        support_arrays = (
            support_y,
            support_x,
            support_profile,
            support_event_probability,
            support_variance,
        )
        if any(
            array.ndim != 1 or array.size != support_size for array in support_arrays
        ):
            raise ValueError("support arrays must be matching 1-D arrays")
        if support_size and (
            np.any((support_y < 0) | (support_y >= rows))
            or np.any((support_x < 0) | (support_x >= columns))
        ):
            raise ValueError("support coordinates fall outside the profile")
        if np.any(~np.isfinite(support_profile)) or np.any(support_profile < 0):
            raise ValueError("support_profile must be finite and nonnegative")
        if np.any(~np.isfinite(support_event_probability)) or np.any(
            (support_event_probability <= 0) | (support_event_probability >= 1)
        ):
            raise ValueError(
                "support_event_probability must be finite and lie in (0, 1)"
            )
        if np.any(~np.isfinite(support_variance)) or np.any(support_variance <= 0):
            raise ValueError(
                "support_detector_variance_adu2 must be finite and positive"
            )
        prior_fraction = float(self.spectrum_prior_fraction)
        prior_floor = float(self.spectrum_prior_floor_adu)
        if not np.isfinite(prior_fraction) or prior_fraction < 0:
            raise ValueError("spectrum_prior_fraction must be finite and nonnegative")
        if not np.isfinite(prior_floor) or prior_floor < 0:
            raise ValueError("spectrum_prior_floor_adu must be finite and nonnegative")

        for array in (
            profile,
            template_spectrum,
            valid_columns,
            support_y,
            support_x,
            support_profile,
            support_event_probability,
            support_variance,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "template_spectrum", template_spectrum)
        object.__setattr__(self, "valid_columns", valid_columns)
        object.__setattr__(self, "support_y", support_y)
        object.__setattr__(self, "support_x", support_x)
        object.__setattr__(self, "support_profile", support_profile)
        object.__setattr__(self, "support_event_probability", support_event_probability)
        object.__setattr__(self, "support_detector_variance_adu2", support_variance)
        object.__setattr__(self, "spectrum_prior_fraction", prior_fraction)
        object.__setattr__(self, "spectrum_prior_floor_adu", prior_floor)


@dataclass(frozen=True)
class ImageConfig:
    """Numerical settings for image inference and guarded correction."""

    gain_e_per_adu: float = 0.7
    saturation_adu: float = 65535.0
    maximum_peak_raw_adu: float = 60000.0
    maximum_iterations: int = 100
    baseline_iterations: int = 6
    convergence_fraction: float = 1.0e-4
    denominator_fraction: float = 1.0e-6
    minimum_information_fraction: float = 0.20
    minimum_data_fraction: float = 0.20
    minimum_finite_center_fraction: float = 0.50
    use_spectrum_reference: bool = False
    gaussian_minimum_expected_electrons: float = 0.0
    poisson_tail_probability: float = 1.0e-12
    poisson_maximum_terms: int = 128
    poisson_chunk_pixels: int = 2048
    seed_residual_z: float = 1.0
    curvature_threshold: float = 2.5
    growth_residual_z: float = 2.25
    ring_radii: tuple[int, ...] = (1, 2)
    ring_minimum_counts: tuple[int, ...] = (3, 8)
    use_diagonal_connectivity: bool = True
    maximum_component_width_columns: int = 10
    uncertainty_inflation: float = 1.0
    frame_offset_mode: str = "zero"
    halo_radius: int = 1
    halo_residual_z: float = 0.0
    halo_conditional_prior: float = 0.5

    def validate(self) -> None:
        positive = (
            "gain_e_per_adu",
            "saturation_adu",
            "maximum_peak_raw_adu",
            "convergence_fraction",
            "denominator_fraction",
            "poisson_tail_probability",
            "seed_residual_z",
            "curvature_threshold",
            "growth_residual_z",
            "uncertainty_inflation",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.gaussian_minimum_expected_electrons)
            or self.gaussian_minimum_expected_electrons < 0
        ):
            raise ValueError(
                "gaussian_minimum_expected_electrons must be finite and nonnegative"
            )
        for name in (
            "minimum_information_fraction",
            "minimum_data_fraction",
            "minimum_finite_center_fraction",
        ):
            value = float(getattr(self, name))
            if not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0, 1]")
        for name in (
            "maximum_iterations",
            "baseline_iterations",
            "poisson_maximum_terms",
            "poisson_chunk_pixels",
            "maximum_component_width_columns",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if len(self.ring_radii) != len(self.ring_minimum_counts):
            raise ValueError("ring radii and minimum counts must have equal length")
        if not self.ring_radii or any(int(value) < 1 for value in self.ring_radii):
            raise ValueError("ring radii must be positive")
        if any(int(value) < 1 for value in self.ring_minimum_counts):
            raise ValueError("ring minimum counts must be positive")
        if self.frame_offset_mode.lower() not in {"zero", "median"}:
            raise ValueError("frame_offset_mode must be 'zero' or 'median'")
        if int(self.halo_radius) < 0:
            raise ValueError("halo_radius must be a nonnegative integer")
        if not np.isfinite(self.halo_residual_z):
            raise ValueError("halo_residual_z must be finite")
        if not 0 <= float(self.halo_conditional_prior) < 1:
            raise ValueError("halo_conditional_prior must lie in [0, 1)")


def build_image_model(
    optical_template: np.ndarray,
    trace_mask: np.ndarray,
    particle_prior: ParticlePrior,
    read_sigma_adu: float | None = None,
    *,
    detector_variance_adu2: np.ndarray | float | None = None,
    spatial_profile_template: np.ndarray | None = None,
    template_spectrum_adu: np.ndarray | None = None,
    spectrum_prior_fraction: float = 0.50,
    spectrum_prior_floor_adu: float | None = None,
    relative_profile_floor: float = 0.002,
) -> ImageModel:
    """Build a reusable image model from independent calibration arrays."""

    optical_template = np.asarray(optical_template, dtype=np.float64)
    trace_mask_input = np.asarray(trace_mask)
    if optical_template.ndim != 2 or optical_template.shape != trace_mask_input.shape:
        raise ValueError("optical_template and trace_mask must be matching 2-D arrays")
    if 0 in optical_template.shape or np.any(~np.isfinite(optical_template)):
        raise ValueError("optical_template must be nonempty and finite")
    if trace_mask_input.dtype.kind not in "buif" or np.any(
        ~np.isfinite(trace_mask_input)
    ):
        raise ValueError("trace_mask must contain finite boolean values")
    if np.any((trace_mask_input != 0) & (trace_mask_input != 1)):
        raise ValueError("trace_mask values must be boolean")
    trace_mask = trace_mask_input.astype(bool, copy=False)
    if not isinstance(particle_prior, ParticlePrior):
        raise TypeError("particle_prior must be a ParticlePrior")
    if optical_template.shape[0] != particle_prior.row_probability.size:
        raise ValueError("particle row prior does not match detector rows")

    if read_sigma_adu is not None:
        read_sigma_adu = float(read_sigma_adu)
        if not np.isfinite(read_sigma_adu) or read_sigma_adu <= 0:
            raise ValueError("read_sigma_adu must be finite and positive")
    spectrum_prior_fraction = float(spectrum_prior_fraction)
    relative_profile_floor = float(relative_profile_floor)
    for name, value in (
        ("spectrum_prior_fraction", spectrum_prior_fraction),
        ("relative_profile_floor", relative_profile_floor),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if spectrum_prior_floor_adu is not None:
        spectrum_prior_floor_adu = float(spectrum_prior_floor_adu)
        if not np.isfinite(spectrum_prior_floor_adu) or spectrum_prior_floor_adu < 0:
            raise ValueError("spectrum_prior_floor_adu must be finite and nonnegative")

    if detector_variance_adu2 is None:
        if read_sigma_adu is None:
            raise ValueError("supply read_sigma_adu or detector_variance_adu2")
        detector_variance = np.full(
            optical_template.shape,
            float(read_sigma_adu) ** 2,
            dtype=np.float64,
        )
    else:
        detector_variance = np.asarray(detector_variance_adu2, dtype=np.float64)
        if detector_variance.ndim == 0:
            detector_variance = np.full(
                optical_template.shape,
                float(detector_variance),
                dtype=np.float64,
            )
        if detector_variance.shape != optical_template.shape:
            raise ValueError("detector_variance_adu2 must be scalar or match the image")
        if np.any(~np.isfinite(detector_variance)) or np.any(detector_variance <= 0):
            raise ValueError("detector_variance_adu2 must be finite and positive")

    representative_read_sigma = float(np.sqrt(np.median(detector_variance)))
    profile_source = (
        optical_template
        if spatial_profile_template is None
        else np.asarray(spatial_profile_template, dtype=np.float64)
    )
    if profile_source.shape != optical_template.shape:
        raise ValueError("spatial_profile_template must match optical_template")
    if np.any(~np.isfinite(profile_source)):
        raise ValueError("spatial_profile_template must be finite")
    positive_profile = np.clip(profile_source, 0.0, None) * trace_mask
    column_maximum = positive_profile.max(axis=0)
    keep = positive_profile >= relative_profile_floor * column_maximum[None, :]
    positive_profile *= keep
    normalization = positive_profile.sum(axis=0)
    valid_columns = normalization > 0
    if not np.any(valid_columns):
        raise ValueError("calibration has no valid trace support")
    profile = np.zeros_like(positive_profile, dtype=np.float64)
    profile[:, valid_columns] = (
        positive_profile[:, valid_columns] / normalization[valid_columns]
    )
    if template_spectrum_adu is None:
        template_spectrum = normalization
    else:
        template_spectrum = np.asarray(template_spectrum_adu, dtype=np.float64)
        if template_spectrum.shape != (optical_template.shape[1],):
            raise ValueError("template_spectrum_adu must have one value per column")
        if np.any(~np.isfinite(template_spectrum)):
            raise ValueError("template_spectrum_adu must be finite")
        template_spectrum = np.clip(template_spectrum, 0.0, None)
        template_spectrum[~valid_columns] = 0.0

    support_mask = profile > 0
    support_y, support_x = np.nonzero(support_mask)
    if spectrum_prior_floor_adu is None:
        spectrum_prior_floor_adu = max(10 * representative_read_sigma, 25.0)
    return ImageModel(
        profile=profile.astype(np.float32),
        template_spectrum=template_spectrum.astype(np.float64),
        valid_columns=valid_columns,
        support_y=support_y.astype(np.int32),
        support_x=support_x.astype(np.int32),
        support_profile=profile[support_y, support_x].astype(np.float64),
        support_event_probability=particle_prior.row_probability[support_y].astype(
            np.float64
        ),
        particle_prior=particle_prior,
        support_detector_variance_adu2=detector_variance[support_y, support_x].astype(
            np.float64
        ),
        spectrum_prior_fraction=float(spectrum_prior_fraction),
        spectrum_prior_floor_adu=float(spectrum_prior_floor_adu),
    )


def _config_fingerprint(config: ImageConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _normal_logpdf(residual: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return -0.5 * (residual / sigma) ** 2 - np.log(sigma) - 0.5 * _LOG_2PI


def _exgaussian_logpdf(
    residual: np.ndarray,
    sigma: np.ndarray,
    exponential_mean: float,
) -> np.ndarray:
    tau = float(exponential_mean)
    return (
        -np.log(tau)
        + 0.5 * (sigma / tau) ** 2
        - residual / tau
        + log_ndtr(residual / sigma - sigma / tau)
    )


def _truncated_charge_mean(
    residual: np.ndarray,
    sigma: np.ndarray,
    exponential_mean: float,
) -> np.ndarray:
    tau = float(exponential_mean)
    shifted = residual - sigma**2 / tau
    alpha = shifted / sigma
    output = np.empty_like(alpha, dtype=np.float64)
    tail = alpha < -8.0
    if np.any(~tail):
        current = alpha[~tail]
        log_ratio = -0.5 * current**2 - 0.5 * _LOG_2PI - log_ndtr(current)
        output[~tail] = shifted[~tail] + sigma[~tail] * np.exp(log_ratio)
    if np.any(tail):
        current = alpha[tail]
        correction = (
            -1.0 / current + 2.0 / current**3 - 10.0 / current**5 + 74.0 / current**7
        )
        output[tail] = sigma[tail] * correction
    negative = output < 0
    tolerance = 1.0e-10 * np.maximum(1.0, np.abs(residual))
    if np.any(output[negative] < -tolerance[negative]):
        raise FloatingPointError("truncated charge mean became materially negative")
    output[negative] = 0.0
    if np.any(~np.isfinite(output)):
        raise FloatingPointError("truncated charge mean is nonfinite")
    return output


def _combine_event_components(
    *,
    clean_log_likelihood: np.ndarray,
    component_log_likelihood: np.ndarray,
    component_charge_mean: np.ndarray,
    event_probability_prior: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior = np.asarray(event_probability_prior, dtype=np.float64)
    joint_clean = np.log1p(-prior) + clean_log_likelihood
    joint_event = (
        np.log(prior)[None, :]
        + np.log(np.clip(np.asarray(weights), 1e-300, None))[:, None]
        + component_log_likelihood
    )
    normalization = logsumexp(np.vstack((joint_clean[None, :], joint_event)), axis=0)
    component_probability = np.exp(joint_event - normalization[None, :])
    event_probability = component_probability.sum(axis=0)
    expected_charge = np.sum(component_probability * component_charge_mean, axis=0)
    return event_probability, expected_charge, component_probability


def _gaussian_event_evidence(
    *,
    residual: np.ndarray,
    sigma: np.ndarray,
    event_probability_prior: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_log = _normal_logpdf(residual, sigma)
    component_log = np.stack(
        [_exgaussian_logpdf(residual, sigma, scale) for scale in scales]
    )
    component_mean = np.stack(
        [_truncated_charge_mean(residual, sigma, scale) for scale in scales]
    )
    return _combine_event_components(
        clean_log_likelihood=clean_log,
        component_log_likelihood=component_log,
        component_charge_mean=component_mean,
        event_probability_prior=event_probability_prior,
        weights=weights,
    )


def _poisson_gaussian_event_evidence(
    *,
    observed_adu: np.ndarray,
    expected_signal_adu: np.ndarray,
    detector_sigma_adu: np.ndarray,
    event_probability_prior: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    config: ImageConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = observed_adu.size
    probability = np.zeros(count, dtype=np.float64)
    expected_charge = np.zeros(count, dtype=np.float64)
    component_probability = np.zeros((scales.size, count), dtype=np.float64)
    gain = float(config.gain_e_per_adu)
    for start in range(0, count, int(config.poisson_chunk_pixels)):
        stop = min(count, start + int(config.poisson_chunk_pixels))
        observed = np.asarray(observed_adu[start:stop], dtype=np.float64)
        expected = np.clip(expected_signal_adu[start:stop], 0.0, None)
        detector_sigma = np.asarray(detector_sigma_adu[start:stop], dtype=np.float64)
        poisson_mean = gain * expected
        maximum_mean = float(np.max(poisson_mean)) if poisson_mean.size else 0.0
        maximum_count = int(np.ceil(maximum_mean + 12 * np.sqrt(maximum_mean + 1) + 12))
        maximum_count = min(maximum_count, int(config.poisson_maximum_terms) - 1)
        counts = np.arange(maximum_count + 1, dtype=np.float64)[:, None]
        log_poisson = (
            xlogy(counts, poisson_mean[None, :])
            - poisson_mean[None, :]
            - gammaln(counts + 1)
        )
        retained_mass = np.exp(logsumexp(log_poisson, axis=0))
        if np.any(1 - retained_mass > float(config.poisson_tail_probability)):
            raise FloatingPointError("Poisson marginalization truncated material mass")
        residual = observed[None, :] - counts / gain
        sigma_matrix = np.broadcast_to(detector_sigma[None, :], residual.shape)
        clean_log = logsumexp(
            log_poisson + _normal_logpdf(residual, sigma_matrix), axis=0
        )
        component_log_rows = []
        component_mean_rows = []
        for scale in scales:
            event_terms = log_poisson + _exgaussian_logpdf(
                residual, sigma_matrix, float(scale)
            )
            component_log_value = logsumexp(event_terms, axis=0)
            conditional_charge = _truncated_charge_mean(
                residual, sigma_matrix, float(scale)
            )
            positive = conditional_charge > 0
            log_charge_terms = np.full_like(event_terms, -np.inf)
            log_charge_terms[positive] = event_terms[positive] + np.log(
                conditional_charge[positive]
            )
            component_charge_value = np.exp(
                logsumexp(log_charge_terms, axis=0) - component_log_value
            )
            component_log_rows.append(component_log_value)
            component_mean_rows.append(component_charge_value)
        current = _combine_event_components(
            clean_log_likelihood=clean_log,
            component_log_likelihood=np.stack(component_log_rows),
            component_charge_mean=np.stack(component_mean_rows),
            event_probability_prior=event_probability_prior[start:stop],
            weights=weights,
        )
        probability[start:stop], expected_charge[start:stop] = current[:2]
        component_probability[:, start:stop] = current[2]
    return probability, expected_charge, component_probability


def _event_evidence(
    *,
    signal_support: np.ndarray,
    optical_mean: np.ndarray,
    detector_variance: np.ndarray,
    event_probability_prior: np.ndarray,
    particle_prior: ParticlePrior,
    config: ImageConfig,
) -> Mapping[str, np.ndarray]:
    expected_electrons = float(config.gain_e_per_adu) * np.clip(optical_mean, 0.0, None)
    exact = expected_electrons < float(config.gaussian_minimum_expected_electrons)
    probability = np.zeros_like(signal_support, dtype=np.float64)
    charge = np.zeros_like(signal_support, dtype=np.float64)
    components = np.zeros(
        (particle_prior.amplitude_scales_adu.size, signal_support.size),
        dtype=np.float64,
    )
    if np.any(~exact):
        variance = detector_variance[~exact] + np.clip(
            optical_mean[~exact], 0.0, None
        ) / float(config.gain_e_per_adu)
        gaussian = _gaussian_event_evidence(
            residual=signal_support[~exact] - optical_mean[~exact],
            sigma=np.sqrt(variance),
            event_probability_prior=event_probability_prior[~exact],
            scales=particle_prior.amplitude_scales_adu,
            weights=particle_prior.amplitude_weights,
        )
        probability[~exact], charge[~exact], components[:, ~exact] = gaussian
    if np.any(exact):
        low_count = _poisson_gaussian_event_evidence(
            observed_adu=signal_support[exact],
            expected_signal_adu=optical_mean[exact],
            detector_sigma_adu=np.sqrt(detector_variance[exact]),
            event_probability_prior=event_probability_prior[exact],
            scales=particle_prior.amplitude_scales_adu,
            weights=particle_prior.amplitude_weights,
            config=config,
        )
        probability[exact], charge[exact], components[:, exact] = low_count
    finite = (
        np.isfinite(probability)
        & np.isfinite(charge)
        & np.all(np.isfinite(components), axis=0)
    )
    return {
        "event_probability": probability,
        "expected_charge": charge,
        "component_probability": components,
        "exact_low_count_branch": exact,
        "finite": finite,
    }


def _column_sum(x: np.ndarray, weights: np.ndarray, n_columns: int) -> np.ndarray:
    return np.bincount(x, weights=weights, minlength=n_columns)


def _solve_spectrum_update(
    *,
    signal_support: np.ndarray,
    profile: np.ndarray,
    support_x: np.ndarray,
    variance: np.ndarray,
    clean_weight: np.ndarray,
    expected_charge: np.ndarray,
    available: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    denominator_fraction: float,
    n_columns: int,
) -> Mapping[str, np.ndarray]:
    valid = np.asarray(available, dtype=bool)
    clean_probability = np.where(valid, np.clip(clean_weight, 0.0, 1.0), 0.0)
    inverse_variance = np.divide(
        1.0,
        variance,
        out=np.zeros_like(variance),
        where=valid & np.isfinite(variance) & (variance > 0),
    )
    nominal_precision = _column_sum(
        support_x, valid * profile**2 * inverse_variance, n_columns
    )
    data_precision = _column_sum(
        support_x, clean_probability * profile**2 * inverse_variance, n_columns
    )
    data_rhs = _column_sum(
        support_x,
        clean_probability
        * profile
        * (signal_support - expected_charge)
        * inverse_variance,
        n_columns,
    )
    denominator = data_precision + prior_precision
    usable = (
        np.isfinite(denominator)
        & np.isfinite(nominal_precision)
        & (nominal_precision > 0)
        & (denominator > float(denominator_fraction) * nominal_precision)
    )
    spectrum = np.maximum(
        0.0,
        np.divide(
            data_rhs + prior_precision * prior_mean,
            denominator,
            out=np.zeros(n_columns, dtype=np.float64),
            where=usable,
        ),
    )
    information_fraction = np.divide(
        data_precision,
        nominal_precision,
        out=np.zeros(n_columns, dtype=np.float64),
        where=nominal_precision > 0,
    )
    data_fraction = np.divide(
        data_precision,
        denominator,
        out=np.zeros(n_columns, dtype=np.float64),
        where=denominator > 0,
    )
    column_variance = np.divide(
        1.0,
        denominator,
        out=np.full(n_columns, np.inf, dtype=np.float64),
        where=usable,
    )
    return {
        "spectrum": spectrum,
        "nominal_precision": nominal_precision,
        "data_precision": data_precision,
        "data_rhs": data_rhs,
        "denominator": denominator,
        "usable": usable,
        "information_fraction": information_fraction,
        "data_fraction": data_fraction,
        "variance": column_variance,
    }


def _reference_terms(
    signal_support: np.ndarray,
    model: ImageModel,
    config: ImageConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = model.support_x
    profile = model.support_profile
    n_columns = model.profile.shape[1]
    initial_variance = model.support_detector_variance_adu2 + np.clip(
        signal_support, 0.0, None
    ) / float(config.gain_e_per_adu)
    initial_precision = _column_sum(x, profile**2 / initial_variance, n_columns)
    initial_rhs = _column_sum(x, profile * signal_support / initial_variance, n_columns)
    initial = np.maximum(
        0.0,
        np.divide(
            initial_rhs,
            initial_precision,
            out=np.zeros(n_columns, dtype=np.float64),
            where=initial_precision > 0,
        ),
    )
    template = np.asarray(model.template_spectrum, dtype=np.float64)
    if not config.use_spectrum_reference:
        return np.zeros_like(template), np.zeros_like(template), initial_variance
    positive = template[np.asarray(model.valid_columns, dtype=bool)]
    cut = float(np.percentile(positive, 70.0)) if positive.size else 0.0
    scale_columns = model.valid_columns & (template >= cut) & (template > 0)
    ratios = initial[scale_columns] / template[scale_columns]
    ratios = ratios[np.isfinite(ratios) & (ratios >= 0)]
    scale = float(np.clip(np.median(ratios), 0.0, 20.0)) if ratios.size else 1.0
    prior_mean = scale * template
    prior_sigma = float(model.spectrum_prior_floor_adu) + float(
        model.spectrum_prior_fraction
    ) * np.abs(prior_mean)
    return prior_mean, 1.0 / prior_sigma**2, initial_variance


def _column_state_masks(
    *,
    invalid: np.ndarray,
    abstained: np.ndarray,
    corrected: np.ndarray,
) -> Mapping[str, np.ndarray]:
    invalid = np.asarray(invalid, dtype=bool)
    abstained = np.asarray(abstained, dtype=bool) & ~invalid
    corrected = np.asarray(corrected, dtype=bool) & ~invalid & ~abstained
    unchanged = ~(invalid | abstained | corrected)
    state = np.full(invalid.size, "unchanged", dtype="U10")
    state[invalid] = "invalid"
    state[abstained] = "abstained"
    state[corrected] = "corrected"
    if not np.all(
        invalid.astype(np.int8)
        + abstained.astype(np.int8)
        + corrected.astype(np.int8)
        + unchanged.astype(np.int8)
        == 1
    ):
        raise AssertionError("final column states are not mutually exclusive")
    return {
        "invalid": invalid,
        "abstain": abstained,
        "corrected": corrected,
        "unchanged": unchanged,
        "state": state,
    }


def _base_inference(
    frame: np.ndarray,
    background: np.ndarray,
    model: ImageModel,
    config: ImageConfig,
) -> Mapping[str, object]:
    frame = np.asarray(frame, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    if frame.shape != background.shape or frame.shape != model.profile.shape:
        raise ValueError("image, background, and calibrated profile must agree")
    y = np.asarray(model.support_y, dtype=np.int64)
    x = np.asarray(model.support_x, dtype=np.int64)
    profile = np.asarray(model.support_profile, dtype=np.float64)
    n_columns = frame.shape[1]
    raw_support = frame[y, x]
    available = (
        np.isfinite(raw_support)
        & np.isfinite(background[y, x])
        & (raw_support < float(config.saturation_adu))
        & np.isfinite(profile)
        & (profile > 0)
    )
    if config.frame_offset_mode.lower() == "zero":
        frame_offset = 0.0
    else:
        difference = frame - background
        finite = difference[np.isfinite(difference)]
        frame_offset = float(np.median(finite)) if finite.size else np.nan
    signal_support = raw_support - background[y, x] - frame_offset
    prior_mean, prior_precision, initial_variance = _reference_terms(
        signal_support, model, config
    )
    result = _solve_spectrum_update(
        signal_support=signal_support,
        profile=profile,
        support_x=x,
        variance=initial_variance,
        clean_weight=np.ones_like(profile),
        expected_charge=np.zeros_like(profile),
        available=available,
        prior_mean=prior_mean,
        prior_precision=prior_precision,
        denominator_fraction=config.denominator_fraction,
        n_columns=n_columns,
    )
    baseline_spectrum = np.asarray(result["spectrum"], dtype=np.float64)
    for _ in range(max(0, int(config.baseline_iterations) - 1)):
        baseline_variance = model.support_detector_variance_adu2 + np.clip(
            profile * baseline_spectrum[x], 0.0, None
        ) / float(config.gain_e_per_adu)
        result = _solve_spectrum_update(
            signal_support=signal_support,
            profile=profile,
            support_x=x,
            variance=baseline_variance,
            clean_weight=np.ones_like(profile),
            expected_charge=np.zeros_like(profile),
            available=available,
            prior_mean=prior_mean,
            prior_precision=prior_precision,
            denominator_fraction=config.denominator_fraction,
            n_columns=n_columns,
        )
        baseline_spectrum = np.asarray(result["spectrum"], dtype=np.float64)

    spectrum = baseline_spectrum.copy()
    event_probability = np.zeros_like(profile)
    expected_charge = np.zeros_like(profile)
    component_probability = np.zeros(
        (model.particle_prior.amplitude_scales_adu.size, profile.size),
        dtype=np.float64,
    )
    converged = np.zeros(n_columns, dtype=bool)
    exact_branch = np.zeros_like(profile, dtype=bool)
    evidence_finite = available.copy()
    iterations = 0
    variance = initial_variance
    for iteration in range(1, int(config.maximum_iterations) + 1):
        iterations = iteration
        optical_mean = profile * spectrum[x]
        variance = model.support_detector_variance_adu2 + np.clip(
            optical_mean, 0.0, None
        ) / float(config.gain_e_per_adu)
        evidence = _event_evidence(
            signal_support=signal_support,
            optical_mean=optical_mean,
            detector_variance=model.support_detector_variance_adu2,
            event_probability_prior=model.support_event_probability,
            particle_prior=model.particle_prior,
            config=config,
        )
        event_probability = np.asarray(evidence["event_probability"])
        expected_charge = np.asarray(evidence["expected_charge"])
        component_probability = np.asarray(evidence["component_probability"])
        exact_branch = np.asarray(evidence["exact_low_count_branch"], dtype=bool)
        evidence_finite = available & np.asarray(evidence["finite"], dtype=bool)
        result = _solve_spectrum_update(
            signal_support=signal_support,
            profile=profile,
            support_x=x,
            variance=variance,
            clean_weight=1.0 - event_probability,
            expected_charge=expected_charge,
            available=evidence_finite,
            prior_mean=prior_mean,
            prior_precision=prior_precision,
            denominator_fraction=config.denominator_fraction,
            n_columns=n_columns,
        )
        updated = np.asarray(result["spectrum"], dtype=np.float64)
        relative = np.abs(updated - spectrum) / np.maximum(1.0, np.abs(spectrum))
        converged = relative <= float(config.convergence_fraction)
        spectrum = updated
        if np.all(converged[np.asarray(model.valid_columns, dtype=bool)]):
            break

    information = np.asarray(result["information_fraction"], dtype=np.float64)
    data_fraction = np.asarray(result["data_fraction"], dtype=np.float64)
    usable = np.asarray(result["usable"], dtype=bool)
    required_columns = np.asarray(model.valid_columns, dtype=bool)
    finite_center_count = _column_sum(x, evidence_finite.astype(float), n_columns)
    support_count = _column_sum(x, np.ones_like(profile), n_columns)
    finite_center_fraction = np.divide(
        finite_center_count,
        support_count,
        out=np.zeros(n_columns),
        where=support_count > 0,
    )
    invalid = ~required_columns
    abstain = required_columns & (
        ~usable
        | ~converged
        | ~np.isfinite(spectrum)
        | (information < float(config.minimum_information_fraction))
        | (data_fraction < float(config.minimum_data_fraction))
        | (finite_center_fraction < float(config.minimum_finite_center_fraction))
    )
    low_count_columns = np.zeros(n_columns, dtype=bool)
    np.logical_or.at(low_count_columns, x, exact_branch & evidence_finite)
    abstain |= required_columns & low_count_columns
    provisional = spectrum.copy()
    provisional[abstain] = baseline_spectrum[abstain]
    return {
        "frame_offset_adu": frame_offset,
        "signal_support": signal_support,
        "available_support": evidence_finite,
        "profile_support": profile,
        "prior_mean": prior_mean,
        "prior_precision": prior_precision,
        "baseline_spectrum": baseline_spectrum,
        "provisional_spectrum": provisional,
        "event_probability": event_probability,
        "expected_charge": expected_charge,
        "component_probability": component_probability,
        "variance_support": variance,
        "solve": result,
        "converged": converged,
        "iterations": iterations,
        "low_count_columns": low_count_columns,
        "invalid_columns": invalid,
        "abstain_columns": abstain,
        "finite_center_fraction": finite_center_fraction,
    }


def _self_excluded_scores(
    base: Mapping[str, object],
    model: ImageModel,
    config: ImageConfig,
) -> Mapping[str, np.ndarray]:
    x = np.asarray(model.support_x, dtype=np.int64)
    profile = np.asarray(base["profile_support"], dtype=np.float64)
    signal = np.asarray(base["signal_support"], dtype=np.float64)
    variance = np.asarray(base["variance_support"], dtype=np.float64)
    available = np.asarray(base["available_support"], dtype=bool)
    probability = np.asarray(base["event_probability"], dtype=np.float64)
    charge = np.asarray(base["expected_charge"], dtype=np.float64)
    solve = base["solve"]
    data_precision = np.asarray(solve["data_precision"], dtype=np.float64)
    nominal_precision = np.asarray(solve["nominal_precision"], dtype=np.float64)
    data_rhs = np.asarray(solve["data_rhs"], dtype=np.float64)
    prior_precision = np.asarray(base["prior_precision"], dtype=np.float64)
    prior_mean = np.asarray(base["prior_mean"], dtype=np.float64)
    clean_probability = np.where(available, 1.0 - probability, 0.0)
    inverse_variance = np.divide(
        1.0,
        variance,
        out=np.zeros_like(variance),
        where=available & np.isfinite(variance) & (variance > 0),
    )
    own_precision = clean_probability * profile**2 * inverse_variance
    own_rhs = clean_probability * profile * (signal - charge) * inverse_variance
    remaining_data = data_precision[x] - own_precision
    denominator = remaining_data + prior_precision[x]
    deletion_available = (
        available
        & np.isfinite(denominator)
        & (denominator > float(config.denominator_fraction) * nominal_precision[x])
        & (nominal_precision[x] > 0)
        & (remaining_data >= -1e-10 * np.maximum(nominal_precision[x], 1.0))
        & (
            remaining_data / np.maximum(nominal_precision[x], 1e-300)
            >= float(config.minimum_information_fraction)
        )
    )
    deleted = np.maximum(
        0.0,
        np.divide(
            data_rhs[x] - own_rhs + prior_precision[x] * prior_mean[x],
            denominator,
            out=np.zeros_like(signal),
            where=deletion_available,
        ),
    )
    residual_variance = (
        model.support_detector_variance_adu2
        + np.clip(profile * deleted, 0.0, None) / float(config.gain_e_per_adu)
        + np.divide(
            profile**2,
            denominator,
            out=np.full_like(profile, np.inf),
            where=deletion_available,
        )
    )
    residual_z = np.divide(
        signal - profile * deleted,
        np.sqrt(residual_variance),
        out=np.full_like(signal, np.nan),
        where=deletion_available & (residual_variance > 0),
    )
    return {
        "deleted_spectrum": deleted,
        "deleted_denominator": denominator,
        "deletion_available": deletion_available,
        "self_excluded_residual_z": residual_z,
    }


def _multiscale_curvature(
    residual_z: np.ndarray,
    available_support: np.ndarray,
    model: ImageModel,
    config: ImageConfig,
) -> Mapping[str, np.ndarray]:
    shape = model.profile.shape
    residual_image = np.full(shape, np.nan, dtype=np.float64)
    available_image = np.zeros(shape, dtype=bool)
    residual_image[model.support_y, model.support_x] = residual_z
    available_image[model.support_y, model.support_x] = available_support
    filled = np.where(available_image, residual_image, 0.0)
    best = np.full(shape, np.nan, dtype=np.float64)
    best_count = np.zeros(shape, dtype=np.int32)
    for radius, minimum in zip(
        config.ring_radii, config.ring_minimum_counts, strict=True
    ):
        axis = np.arange(-int(radius), int(radius) + 1)
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        kernel = (np.maximum(np.abs(yy), np.abs(xx)) == int(radius)).astype(float)
        count = ndi.convolve(
            available_image.astype(float), kernel, mode="constant", cval=0.0
        )
        total = ndi.convolve(filled, kernel, mode="constant", cval=0.0)
        valid = available_image & (count >= int(minimum))
        neighbor_mean = np.divide(
            total, count, out=np.zeros_like(total), where=count > 0
        )
        denominator = np.sqrt(
            1.0 + np.divide(1.0, count, out=np.zeros_like(count), where=count > 0)
        )
        score = np.divide(
            residual_image - neighbor_mean,
            denominator,
            out=np.full(shape, np.nan),
            where=valid,
        )
        replace = valid & (~np.isfinite(best) | (score > best))
        best[replace] = score[replace]
        best_count[replace] = count[replace].astype(np.int32)
    return {
        "curvature_support": best[model.support_y, model.support_x],
        "ring_count_support": best_count[model.support_y, model.support_x],
    }


def _profile_extract(
    frame: np.ndarray,
    background: np.ndarray,
    model: ImageModel,
    config: ImageConfig,
) -> Mapping[str, object]:
    base = _base_inference(frame, background, model, config)
    n_columns = model.profile.shape[1]
    x = np.asarray(model.support_x, dtype=np.int64)
    invalid = np.asarray(base["invalid_columns"], dtype=bool).copy()
    abstain = np.asarray(base["abstain_columns"], dtype=bool).copy()
    eligible_support = (
        np.asarray(base["available_support"], dtype=bool) & ~invalid[x] & ~abstain[x]
    )

    deleted = _self_excluded_scores(base, model, config)
    deletion_available = (
        np.asarray(deleted["deletion_available"], dtype=bool) & eligible_support
    )
    curvature = _multiscale_curvature(
        np.asarray(deleted["self_excluded_residual_z"], dtype=float),
        deletion_available,
        model,
        config,
    )
    residual_z = np.asarray(deleted["self_excluded_residual_z"], dtype=float)
    curvature_score = np.asarray(curvature["curvature_support"], dtype=float)
    finite_center_count = _column_sum(
        x,
        (deletion_available & np.isfinite(curvature_score)).astype(float),
        n_columns,
    )
    support_count = _column_sum(x, np.ones_like(residual_z), n_columns)
    center_coverage = np.divide(
        finite_center_count,
        support_count,
        out=np.zeros(n_columns),
        where=support_count > 0,
    )
    abstain |= (~invalid) & (
        center_coverage < float(config.minimum_finite_center_fraction)
    )

    seed_support = (
        deletion_available
        & np.isfinite(curvature_score)
        & (residual_z > float(config.seed_residual_z))
        & (curvature_score > float(config.curvature_threshold))
    )
    grow_support = deletion_available & (residual_z >= float(config.growth_residual_z))
    seed_image = np.zeros(model.profile.shape, dtype=bool)
    grow_image = np.zeros(model.profile.shape, dtype=bool)
    seed_image[model.support_y, model.support_x] = seed_support
    grow_image[model.support_y, model.support_x] = grow_support | seed_support
    structure = (
        np.ones((3, 3), dtype=bool)
        if config.use_diagonal_connectivity
        else ndi.generate_binary_structure(2, 1)
    )
    grown = ndi.binary_propagation(seed_image, structure=structure, mask=grow_image)
    labels, count = ndi.label(grown, structure=structure)

    accepted_image = np.zeros_like(grown)
    out_of_model_image = np.zeros_like(grown)
    raw = np.asarray(frame, dtype=float)
    maximum_pixels = model.particle_prior.maximum_event_pixels
    maximum_pixels = (
        int(maximum_pixels) if maximum_pixels is not None else np.iinfo(np.int32).max
    )
    for label_id in range(1, int(count) + 1):
        component = labels == label_id
        yy, xx = np.nonzero(component)
        if yy.size == 0:
            continue
        dilation = ndi.binary_dilation(component, structure=np.ones((3, 3), dtype=bool))
        touches_boundary = bool(
            np.any(yy == 0)
            or np.any(yy == raw.shape[0] - 1)
            or np.any(xx == 0)
            or np.any(xx == raw.shape[1] - 1)
        )
        violates = (
            not np.any(component & seed_image)
            or yy.size > maximum_pixels
            or xx.max() - xx.min() + 1 > int(config.maximum_component_width_columns)
            or touches_boundary
            or bool(np.any(raw[dilation] >= float(config.maximum_peak_raw_adu)))
        )
        if violates:
            out_of_model_image |= component
        else:
            accepted_image |= component

    abstain |= np.any(out_of_model_image, axis=0) & ~invalid
    accepted_support = accepted_image[model.support_y, model.support_x]
    accepted_support &= ~abstain[x]
    probability = np.asarray(base["event_probability"], dtype=np.float64).copy()
    expected_charge = np.asarray(base["expected_charge"], dtype=np.float64).copy()

    halo_image = np.zeros_like(accepted_image)
    if int(config.halo_radius) > 0 and np.any(accepted_image):
        dilated = accepted_image.copy()
        for _ in range(int(config.halo_radius)):
            dilated = ndi.binary_dilation(dilated, structure=structure)
        halo_image = dilated & ~accepted_image
    halo_support = halo_image[model.support_y, model.support_x]
    halo_support &= (
        np.asarray(base["available_support"], dtype=bool)
        & ~abstain[x]
        & np.isfinite(residual_z)
        & (residual_z > float(config.halo_residual_z))
    )
    if np.any(halo_support):
        conditional = np.asarray(
            model.support_event_probability, dtype=np.float64
        ).copy()
        conditional[halo_support] = np.maximum(
            conditional[halo_support], float(config.halo_conditional_prior)
        )
        revised = _event_evidence(
            signal_support=np.asarray(base["signal_support"], dtype=np.float64),
            optical_mean=np.asarray(base["profile_support"], dtype=np.float64)
            * np.asarray(base["solve"]["spectrum"], dtype=np.float64)[x],
            detector_variance=model.support_detector_variance_adu2,
            event_probability_prior=conditional,
            particle_prior=model.particle_prior,
            config=config,
        )
        update = halo_support & np.asarray(revised["finite"], dtype=bool)
        probability = np.where(
            update, np.asarray(revised["event_probability"], dtype=float), probability
        )
        expected_charge = np.where(
            update, np.asarray(revised["expected_charge"], dtype=float), expected_charge
        )

    clean_weight = 1.0 - probability
    clean_weight[accepted_support] = 0.0
    refit = _solve_spectrum_update(
        signal_support=np.asarray(base["signal_support"], dtype=float),
        profile=np.asarray(base["profile_support"], dtype=float),
        support_x=x,
        variance=np.asarray(base["variance_support"], dtype=float),
        clean_weight=clean_weight,
        expected_charge=expected_charge,
        available=np.asarray(base["available_support"], dtype=bool),
        prior_mean=np.asarray(base["prior_mean"], dtype=float),
        prior_precision=np.asarray(base["prior_precision"], dtype=float),
        denominator_fraction=config.denominator_fraction,
        n_columns=n_columns,
    )

    if np.any(accepted_support):
        target = accepted_support | halo_support
        conditional = np.asarray(
            model.support_event_probability, dtype=np.float64
        ).copy()
        conditional[target] = np.maximum(
            conditional[target], float(config.halo_conditional_prior)
        )
        revised = _event_evidence(
            signal_support=np.asarray(base["signal_support"], dtype=np.float64),
            optical_mean=np.asarray(base["profile_support"], dtype=np.float64)
            * np.asarray(refit["spectrum"], dtype=np.float64)[x],
            detector_variance=model.support_detector_variance_adu2,
            event_probability_prior=conditional,
            particle_prior=model.particle_prior,
            config=config,
        )
        update = target & np.asarray(revised["finite"], dtype=bool)
        probability = np.where(
            update, np.asarray(revised["event_probability"], dtype=float), probability
        )
        expected_charge = np.where(
            update, np.asarray(revised["expected_charge"], dtype=float), expected_charge
        )

    corrected = np.zeros(n_columns, dtype=bool)
    np.logical_or.at(corrected, x, accepted_support)
    refit_ok = (
        np.asarray(refit["usable"], dtype=bool)
        & (
            np.asarray(refit["information_fraction"], dtype=float)
            >= float(config.minimum_information_fraction)
        )
        & (
            np.asarray(refit["data_fraction"], dtype=float)
            >= float(config.minimum_data_fraction)
        )
        & np.isfinite(np.asarray(refit["spectrum"], dtype=float))
    )
    abstain |= corrected & ~refit_ok
    corrected &= refit_ok & ~invalid & ~abstain
    states = _column_state_masks(
        invalid=invalid, abstained=abstain, corrected=corrected
    )
    baseline = np.asarray(base["baseline_spectrum"], dtype=float)
    spectrum = baseline.copy()
    spectrum[corrected] = np.asarray(refit["spectrum"], dtype=float)[corrected]
    spectrum_variance = np.full(n_columns, np.inf, dtype=float)
    baseline_variance = np.asarray(base["solve"]["variance"], dtype=float)
    spectrum_variance[states["unchanged"]] = baseline_variance[states["unchanged"]]
    spectrum_variance[corrected] = (
        np.asarray(refit["variance"], dtype=float)[corrected]
        * float(config.uncertainty_inflation) ** 2
    )
    accepted_support &= corrected[x]
    return {
        **base,
        **states,
        "spectrum": spectrum,
        "spectrum_variance": spectrum_variance,
        "accepted_support": accepted_support,
        "halo_support": halo_support,
        "out_of_model_support": out_of_model_image[model.support_y, model.support_x],
        "self_excluded_residual_z": residual_z,
        "curvature_score": curvature_score,
        "ring_count": curvature["ring_count_support"],
        "center_coverage": center_coverage,
        "seed_support": seed_support,
        "event_probability": probability,
        "expected_charge": expected_charge,
        "refit": refit,
        "configuration_sha256": _config_fingerprint(config),
    }


def clean_image(
    image: np.ndarray,
    background: np.ndarray,
    model: ImageModel,
    *,
    config: ImageConfig | None = None,
    repair: str = "model",
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, Mapping[str, object]]:
    """Remove accepted cosmic-ray charge from one calibrated detector image.

    ``background`` is the per-pixel dark plus instrumental background map.
    ``repair='model'`` replaces accepted event pixels with the clean profile
    refit. ``repair='subtract'`` subtracts posterior expected charge instead.
    Abstained, unsupported, and uncorrected columns remain unchanged. Image and
    background arrays must contain only finite values.
    """

    if repair not in {"model", "subtract"}:
        raise ValueError("repair must be 'model' or 'subtract'")
    if not isinstance(model, ImageModel):
        raise TypeError("model must be an ImageModel")
    if config is None:
        config = ImageConfig()
    elif not isinstance(config, ImageConfig):
        raise TypeError("config must be an ImageConfig")
    config.validate()
    if (
        config.use_spectrum_reference
        and model.spectrum_prior_fraction <= 0.0
        and model.spectrum_prior_floor_adu <= 0.0
    ):
        raise ValueError(
            "the spectrum-reference fraction or floor must be positive when enabled"
        )
    image = np.asarray(image, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    if image.shape != model.profile.shape or background.shape != image.shape:
        raise ValueError("image, background, and calibrated profile must agree")
    if np.any(~np.isfinite(image)) or np.any(~np.isfinite(background)):
        raise ValueError("image and background must be finite")

    result = _profile_extract(image, background, model, config)
    y = np.asarray(model.support_y, dtype=np.int64)
    x = np.asarray(model.support_x, dtype=np.int64)
    corrected_columns = np.asarray(result["corrected"], dtype=bool)
    expected_charge = np.asarray(result["expected_charge"], dtype=np.float64)
    accepted = np.asarray(result["accepted_support"], dtype=bool)
    event_charge = np.where(
        corrected_columns[x] & np.isfinite(expected_charge), expected_charge, 0.0
    )
    cleaned = image.copy()
    cleaned[y, x] -= event_charge

    if repair == "model":
        spectrum = np.asarray(result["spectrum"], dtype=np.float64)
        fitted = (
            background[y, x]
            + float(result["frame_offset_adu"])
            + np.asarray(model.support_profile, dtype=np.float64) * spectrum[x]
        )
        replace = accepted & np.isfinite(fitted)
        cleaned[y[replace], x[replace]] = fitted[replace]

    if not return_details:
        return cleaned
    details = dict(result)
    details["cleaned_image"] = cleaned
    details["removed_charge_adu"] = float(np.sum(image - cleaned))
    details["repair"] = repair
    return cleaned, details

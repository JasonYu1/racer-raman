"""Censored hyperexponential charge prior for detector-image events.

The event-pixel catalogue is left truncated: a pixel only enters it when its
residual clears the growth threshold, so the smallest charges are missing in
a pixel-dependent way.  Fitting the raw charges therefore biases every scale.
These helpers fit the underlying positive charge distribution conditioned on
that per-pixel threshold, which is what Algorithm S1a step 6 specifies.  The
result stays a finite mixture of exponentials, so the Gaussian convolution in
the inference step remains analytic.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

__all__ = [
    "fit_charge_prior",
    "charge_prior_loglikelihood",
    "select_charge_prior",
]


def _validated_observations(
    charge_adu: np.ndarray,
    threshold_adu: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    charge = np.asarray(charge_adu, dtype=np.float64)
    if threshold_adu is None:
        threshold = np.zeros_like(charge)
    else:
        threshold = np.asarray(threshold_adu, dtype=np.float64)
    if charge.ndim != 1 or threshold.shape != charge.shape:
        raise ValueError("charge and threshold must be matching 1-D arrays")
    if not charge.size:
        raise ValueError("at least one charge observation is required")
    if np.any(~np.isfinite(charge)) or np.any(~np.isfinite(threshold)):
        raise ValueError("charge and threshold observations must be finite")
    if np.any(charge <= 0) or np.any(threshold < 0) or np.any(charge < threshold):
        raise ValueError(
            "charges must be positive and no smaller than their thresholds"
        )
    return charge, threshold


def _validated_mixture(
    scales_adu: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scales = np.asarray(scales_adu, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if scales.ndim != 1 or weights.shape != scales.shape or not scales.size:
        raise ValueError("scales and weights must be nonempty matching 1-D arrays")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("all exponential scales must be finite and positive")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative and sum to more than zero")
    return scales, weights / weights.sum()


def charge_prior_loglikelihood(
    charge_adu: np.ndarray,
    threshold_adu: np.ndarray | None,
    scales_adu: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Per-pixel log likelihood of the threshold-conditioned density.

    This is ``log f(c) - log S(T)`` with ``f`` the mixture density and ``S``
    its survival function, i.e. the denominator of Algorithm S1a step 6.
    """

    charge, threshold = _validated_observations(charge_adu, threshold_adu)
    scales, weights = _validated_mixture(scales_adu, weights)
    log_weight = np.log(np.clip(weights, 1e-300, None))[:, None]
    log_scale = np.log(scales)[:, None]
    log_survival_at_threshold = logsumexp(
        log_weight - threshold[None, :] / scales[:, None], axis=0
    )
    log_density_at_charge = logsumexp(
        log_weight - log_scale - charge[None, :] / scales[:, None], axis=0
    )
    return log_density_at_charge - log_survival_at_threshold


def fit_charge_prior(
    charge_adu: np.ndarray,
    threshold_adu: np.ndarray | None = None,
    *,
    n_components: int,
    random_seed: int = 0,
    n_starts: int = 8,
    maximum_iterations: int = 1500,
) -> Mapping[str, object]:
    """Fit the mixture by multistart maximum likelihood under truncation."""

    charge, threshold = _validated_observations(charge_adu, threshold_adu)
    n_components = int(n_components)
    if n_components < 1:
        raise ValueError("n_components must be positive")
    n_starts = max(1, int(n_starts))
    rng = np.random.default_rng(random_seed)

    probability_grid = np.linspace(0.12, 0.92, n_components)
    base_scales = np.quantile(charge, probability_grid) / np.log(2.0)
    minimum_scale = max(0.25 * float(np.median(threshold)), 0.25)
    maximum_scale = max(20.0 * float(np.max(charge)), 100.0 * minimum_scale)
    base_scales = np.clip(base_scales, minimum_scale, maximum_scale)

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_scales = parameters[:n_components]
        if n_components == 1:
            mixture_weights = np.ones(1, dtype=np.float64)
        else:
            logits = np.concatenate(
                (parameters[n_components:], np.zeros(1, dtype=np.float64))
            )
            mixture_weights = np.exp(logits - logsumexp(logits))
        return np.exp(log_scales), mixture_weights

    def objective(parameters: np.ndarray) -> float:
        scales_here, weights_here = unpack(parameters)
        return float(
            -np.sum(
                charge_prior_loglikelihood(
                    charge, threshold, scales_here, weights_here
                )
            )
        )

    bounds = [(np.log(minimum_scale), np.log(maximum_scale))] * n_components
    bounds += [(-18.0, 18.0)] * max(0, n_components - 1)
    best = None
    for start in range(n_starts):
        if start == 0:
            initial_scales = base_scales.copy()
            initial_weights = np.full(n_components, 1.0 / n_components)
        else:
            initial_scales = np.clip(
                base_scales * np.exp(rng.normal(0.0, 0.9, n_components)),
                minimum_scale,
                maximum_scale,
            )
            initial_weights = rng.dirichlet(np.ones(n_components))
        if n_components == 1:
            initial = np.log(initial_scales)
        else:
            logits = np.log(initial_weights[:-1]) - np.log(initial_weights[-1])
            initial = np.concatenate((np.log(initial_scales), logits))
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maximum_iterations), "ftol": 1e-10},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result

    if best is None:  # pragma: no cover - minimize always returns a result
        raise RuntimeError("charge-prior optimizer did not run")
    scales, weights = unpack(best.x)
    order = np.argsort(scales)
    log_likelihood = -float(best.fun)
    n_parameters = 2 * n_components - 1
    return {
        "scales_adu": scales[order],
        "weights": weights[order],
        "log_likelihood": log_likelihood,
        "bic": float(np.log(charge.size) * n_parameters - 2.0 * log_likelihood),
        "n_components": n_components,
        "n_observations": int(charge.size),
        "success": bool(best.success),
    }


def select_charge_prior(
    charge_adu: np.ndarray,
    threshold_adu: np.ndarray,
    inner_fit: np.ndarray,
    *,
    maximum_components: int = 5,
    random_seed: int = 0,
    minimum_validation_gain: float = 0.01,
) -> Mapping[str, object]:
    """Choose the component count on held-out pixels, then refit on all of them.

    Candidates are fitted on the inner-fit partition and scored by mean
    per-pixel log likelihood on the inner-validation partition.  The selected
    count is the smallest ``K >= 2`` whose gain from adding one more component
    falls below ``minimum_validation_gain``, and otherwise the best-scoring
    count.  The returned mixture is refitted on every pixel at that count, as
    Algorithm S1a step 6 specifies.
    """

    charge, threshold = _validated_observations(charge_adu, threshold_adu)
    inner_fit = np.asarray(inner_fit, dtype=bool)
    if inner_fit.shape != charge.shape:
        raise ValueError("inner_fit must match the charge observations")
    if not inner_fit.any() or inner_fit.all():
        raise ValueError(
            "component selection needs event pixels in both inner partitions"
        )

    scores: list[float] = []
    for components in range(1, int(maximum_components) + 1):
        candidate = fit_charge_prior(
            charge[inner_fit],
            threshold[inner_fit],
            n_components=components,
            random_seed=int(random_seed) + 1000 + components,
            n_starts=6,
        )
        scores.append(
            float(
                np.mean(
                    charge_prior_loglikelihood(
                        charge[~inner_fit],
                        threshold[~inner_fit],
                        candidate["scales_adu"],
                        candidate["weights"],
                    )
                )
            )
        )

    gains = np.diff(np.asarray(scores, dtype=np.float64))
    selected = 0
    for index in range(1, len(scores) - 1):
        if gains[index] < float(minimum_validation_gain):
            selected = index + 1
            break
    if selected == 0:
        selected = int(np.argmax(scores)) + 1

    final = fit_charge_prior(
        charge,
        threshold,
        n_components=selected,
        random_seed=int(random_seed) + 2000,
        n_starts=10,
    )
    return {
        **final,
        "selected_components": int(selected),
        "validation_mean_log_likelihood": tuple(scores),
    }

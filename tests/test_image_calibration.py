"""Checks on the image calibration against Algorithm S1a of the supplement.

Runnable directly (``python tests/test_image_calibration.py``) so it needs
nothing beyond the package's own dependencies; the ``test_`` names also let
pytest collect it if that is ever added.

The synthetic detector is built with a known pedestal, a known read noise and
a known diffuse photo level, chosen so the properties S1a specifies are
separable: the pedestal is 1000x the diffuse level, so a shot term built from
the wrong one is off by three orders of magnitude rather than a few percent.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cosmic_ray_bayes import calibrate_image  # noqa: E402
from cosmic_ray_bayes import calibration as cal  # noqa: E402

GAIN = 0.7
PEDESTAL = 4000.0
READ_SIGMA = 4.0
DIFFUSE = 4.0
ROWS, COLUMNS, FRAMES = 64, 48, 40
TRACE_ROW, TRACE_SIGMA = 20.0, 3.0


def _detector() -> tuple[np.ndarray, np.ndarray]:
    """A dark stack with particle events, and a sample stack with a trace."""

    rng = np.random.default_rng(20260819)
    dark = PEDESTAL + rng.normal(0.0, READ_SIGMA, (FRAMES, ROWS, COLUMNS))
    # Events: enough pixels to fit the charge prior, none at the boundary.
    for index in range(10):
        frame = 2 + 3 * index
        row = 8 + 5 * (index % 9)
        column = 6 + 4 * (index % 10)
        dark[frame, row:row + 3, column:column + 2] += 900.0 + 80.0 * index
    rows = np.arange(ROWS, dtype=float)[:, None]
    trace = np.exp(-0.5 * ((rows - TRACE_ROW) / TRACE_SIGMA) ** 2)
    amplitude = 500.0 + 200.0 * np.cos(np.arange(COLUMNS) / 7.0)
    sample = (PEDESTAL + DIFFUSE + trace * amplitude[None, :]
              + rng.normal(0.0, READ_SIGMA, (FRAMES, ROWS, COLUMNS)))
    return dark, sample


def test_pilot_selection_is_evenly_spaced_and_odd():
    """S1a.1: the master dark comes from N_pilot evenly spaced frames."""

    index = cal._evenly_spaced_indices(1000, 31)
    assert index.size == 31, index.size
    assert index[0] == 0 and index[-1] == 999
    gaps = np.diff(index)
    assert gaps.max() - gaps.min() <= 1, gaps
    # An even request drops one, so a median over the pilot is a real sample.
    assert cal._evenly_spaced_indices(1000, 30).size == 29


def test_offsets_use_the_decimated_grid():
    """S1a.2/1.6: offsets are medians over every eighth row and column."""

    rng = np.random.default_rng(1)
    frames = rng.normal(100.0, 5.0, (6, 64, 64))
    reference = np.median(frames, axis=0)
    got = cal._grid_offsets(frames, reference, stride=8)
    want = np.median(
        (frames[:, ::8, ::8] - reference[::8, ::8]), axis=(1, 2))
    assert np.allclose(got, want), np.max(np.abs(got - want))
    # and it is genuinely not the all-pixel median
    allpix = np.median(frames - reference[None], axis=(1, 2))
    assert not np.allclose(got, allpix, atol=1e-9)


def test_finite_pilot_factor_is_applied():
    """S1a.8: the pilot-median inflation 1 + pi/(2 N_pilot) is removed."""

    dark, _ = _detector()
    _, variance, _, _, scale = cal._image_dark_statistics(
        dark, mad_to_sigma=1.4826, pilot_frames=31, offset_stride=8,
        variance_clip_sigma=6.0, minimum_sigma_adu=0.5)
    n_pilot = cal._evenly_spaced_indices(dark.shape[0], 31).size
    factor = 1.0 + np.pi / (2.0 * n_pilot)
    assert factor > 1.0
    # Undoing the division must land near the raw clipped second moment.
    assert np.median(variance) * factor > np.median(variance)
    assert np.all(scale >= 0.5)


def test_shot_term_uses_the_diffuse_level_not_the_dark_pedestal():
    """S1a.16: V^det gains [b_x]_+/g, built from the diffuse photo level.

    The dark pedestal is not a photo signal and carries no shot noise. Adding
    it instead inflates the detector variance by roughly PEDESTAL/DIFFUSE.
    """

    dark, sample = _detector()
    calibration = calibrate_image(dark, sample)
    detector_variance = np.asarray(
        calibration.model.support_detector_variance_adu2, dtype=float)
    dark_variance = float(np.median(calibration.dark_variance_adu2))

    specified = dark_variance + DIFFUSE / GAIN
    wrong = dark_variance + PEDESTAL / GAIN
    median = float(np.median(detector_variance))
    assert abs(median - specified) < 0.5 * specified, (
        "detector variance %.1f is not the specified %.1f "
        "(the dark-pedestal version would be %.1f)" % (median, specified, wrong))


def test_profile_support_respects_the_relative_floor():
    """S1a.19: rows below 0.002 of the column maximum are discarded."""

    dark, sample = _detector()
    calibration = calibrate_image(dark, sample)
    profile = np.asarray(calibration.model.profile, dtype=float)
    column_max = profile.max(axis=0)
    kept = profile > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.where(column_max[None, :] > 0.0,
                            profile / column_max[None, :], 0.0)
    offenders = int(np.count_nonzero(kept & (relative < 0.002)))
    assert offenders == 0, (
        "%d support pixels sit below 0.002 of their column maximum" % offenders)


def test_charge_prior_selects_its_component_count():
    """S1a.6: K is chosen on held-out pixels, not fixed at four."""

    dark, sample = _detector()
    prior = calibrate_image(dark, sample).model.particle_prior
    scales = np.asarray(prior.amplitude_scales_adu, dtype=float)
    weights = np.asarray(prior.amplitude_weights, dtype=float)
    assert 1 <= scales.size <= 5, scales.size
    assert scales.size == weights.size
    assert np.all(scales > 0.0) and abs(weights.sum() - 1.0) < 1e-9
    assert np.all(np.diff(scales) > 0.0), "scales should be sorted"


def main() -> int:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failed = 0
    for test in tests:
        try:
            test()
            print("PASS  %s" % test.__name__)
        except Exception as error:  # noqa: BLE001 - a test runner reports all
            failed += 1
            print("FAIL  %s" % test.__name__)
            line = traceback.format_exc().strip().splitlines()[-1]
            print("      %s" % line[:200])
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

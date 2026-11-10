"""Checks on the row-summed reference calibration and its gap filling.

Runnable directly (``python tests/test_spectrum_reference.py``) so it needs
nothing beyond the package's own dependencies; the ``test_`` names also let
pytest collect it if that is ever added.

The subject is ``_fill_missing``, which supplies a level for columns that no
frame of a stack can measure -- saturated or nonfinite in every frame. Two
orders are possible and they disagree: interpolate from the stack's own
finite neighbours and fall back only for what remains, or substitute the
fallback first and interpolate only what the fallback cannot cover. The
reference implementation in this file is the first order, which is what the
method of record uses; the fallback is the closed-shutter dark, whose
pedestal is far below an open-shutter level, so the two orders differ by the
full signal height on exactly the columns that are hardest to measure.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cosmic_ray_bayes import calibration as cal  # noqa: E402
from cosmic_ray_bayes.spectrum import build_spectrum_calibration  # noqa: E402

COLUMNS = 64
FRAMES = 40
SATURATION = 65535.0
DARK_PEDESTAL = 500.0
OPEN_LEVEL = 20000.0


def _record_order(
    values: np.ndarray,
    fallback: np.ndarray | None,
) -> np.ndarray:
    """Interpolate from finite neighbours first, fall back for the rest."""

    result = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(result)
    if np.any(finite):
        missing = np.flatnonzero(~finite)
        if missing.size:
            good = np.flatnonzero(finite)
            result[missing] = np.interp(missing, good, result[good])
    elif fallback is not None:
        result = np.asarray(fallback, dtype=float).copy()
    if fallback is not None:
        fallback_values = np.asarray(fallback, dtype=float)
        replace = ~np.isfinite(result) & np.isfinite(fallback_values)
        result[replace] = fallback_values[replace]
    return result


def _token_particle_prior() -> "object":
    """A minimal valid particle prior, so the dark container can be built.

    The variance floor under test does not consult it; it exists only because
    SpectrumDarkCalibration requires one.
    """

    rng = np.random.default_rng(3)
    return build_spectrum_calibration(
        rng.exponential(400.0, 256),
        rng.integers(1, 5, 128),
        events_per_frame=0.5,
        detector_columns=COLUMNS,
        exposure_ms=1000.0,
    )


def test_fill_interpolates_before_consulting_the_fallback() -> None:
    values = np.array([10.0, np.nan, 12.0, np.nan, 14.0])
    fallback = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    filled = cal._fill_missing(values, fallback)
    expected = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    if not np.array_equal(filled, expected):
        raise AssertionError(
            "fallback consulted before interpolation: got %s, expected %s"
            % (filled, expected)
        )


def test_fill_matches_the_reference_order_bit_for_bit() -> None:
    rng = np.random.default_rng(11)
    for trial in range(400):
        size = int(rng.integers(3, 40))
        values = rng.normal(OPEN_LEVEL, 20.0, size)
        values[rng.random(size) < 0.3] = np.nan
        fallback = rng.normal(DARK_PEDESTAL, 5.0, size)
        if rng.random() < 0.2:
            fallback[rng.random(size) < 0.3] = np.nan
        if rng.random() < 0.25:
            fallback = None
        if not np.any(np.isfinite(values)) and fallback is None:
            continue  # no usable level exists; _fill_missing refuses by design
        expected = _record_order(values.copy(),
                                 None if fallback is None else fallback.copy())
        filled = cal._fill_missing(values.copy(),
                                   None if fallback is None else fallback.copy())
        if not np.array_equal(filled, expected):
            raise AssertionError(
                "trial %d diverged from the reference order:\n"
                "  values   %s\n  fallback %s\n  expected %s\n  got      %s"
                % (trial, values, fallback, expected, filled)
            )


def test_fill_leaves_a_complete_stack_untouched() -> None:
    values = np.linspace(100.0, 200.0, 16)
    filled = cal._fill_missing(values, np.zeros(16))
    if not np.array_equal(filled, values):
        raise AssertionError("a stack with no gaps was modified")


def test_fill_uses_the_fallback_when_no_column_is_usable() -> None:
    values = np.full(8, np.nan)
    fallback = np.arange(8, dtype=float) + DARK_PEDESTAL
    filled = cal._fill_missing(values, fallback)
    if not np.array_equal(filled, fallback):
        raise AssertionError("the fallback was not used for a dead stack")


def test_fill_refuses_a_dead_stack_with_no_fallback() -> None:
    try:
        cal._fill_missing(np.full(8, np.nan), None)
    except ValueError:
        return
    raise AssertionError("a dead stack with no fallback was filled anyway")


def test_reference_master_interpolates_an_always_saturated_column() -> None:
    """The reference master must not inherit the dark pedestal.

    Column 32 is saturated in every frame of the open-shutter stack, so no
    frame measures it. Its neighbours sit at the open level, roughly forty
    times the dark pedestal, which makes the two fill orders trivially
    distinguishable.
    """

    rng = np.random.default_rng(5)
    dark = np.full((FRAMES, COLUMNS), DARK_PEDESTAL)
    dark += rng.normal(0.0, 3.0, (FRAMES, COLUMNS))
    dark_calibration = cal.estimate_temporal_statistics(
        dark,
        saturation_adu=SATURATION,
    )

    reference = np.full((FRAMES, COLUMNS), OPEN_LEVEL)
    reference += rng.normal(0.0, 8.0, (FRAMES, COLUMNS))
    reference[:, 32] = SATURATION
    filled = cal.estimate_temporal_statistics(
        reference,
        fallback=dark_calibration,
        saturation_adu=SATURATION,
    )

    level = float(filled.master_adu[32])
    neighbourhood = float(np.median(filled.master_adu[[30, 31, 33, 34]]))
    if abs(level - neighbourhood) > 200.0:
        raise AssertionError(
            "the always-saturated column took %.1f ADU while its neighbours "
            "sit at %.1f ADU; the dark pedestal is %.1f"
            % (level, neighbourhood, DARK_PEDESTAL)
        )
    if abs(level - DARK_PEDESTAL) < 1000.0:
        raise AssertionError(
            "the always-saturated column inherited the dark pedestal (%.1f "
            "ADU) instead of its neighbours' level" % level
        )


def test_reference_variance_is_floored_by_the_dark() -> None:
    rng = np.random.default_rng(7)
    dark = np.full((FRAMES, COLUMNS), DARK_PEDESTAL)
    dark += rng.normal(0.0, 30.0, (FRAMES, COLUMNS))
    dark_calibration = cal.estimate_temporal_statistics(
        dark,
        saturation_adu=SATURATION,
    )
    quiet = np.full((FRAMES, COLUMNS), OPEN_LEVEL)
    quiet += rng.normal(0.0, 1.0, (FRAMES, COLUMNS))
    measured = cal.calibrate_spectrum_reference(
        quiet,
        cal.SpectrumDarkCalibration(
            temporal=dark_calibration,
            particle_calibration=_token_particle_prior(),
        ),
        saturation_adu=SATURATION,
    )
    below = measured.variance_adu2 < dark_calibration.variance_adu2 - 1e-9
    if np.any(below):
        raise AssertionError(
            "%d columns fell below the dark variance floor"
            % int(np.count_nonzero(below))
        )


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

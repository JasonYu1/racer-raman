# RACER

Readout-aware cosmic-ray event removal for Raman spectroscopy.

Two methods, one per detector readout mode:

- **RACER-2D** corrects two-dimensional detector frames. It tests each pixel
  against the calibrated optical row profile, so a particle is separated from
  Raman signal by where its charge sits across the detector rows.
- **RACER-1D** corrects row-summed (full vertical binning) spectra, where the
  row coordinate has been summed away. It relies instead on measured charge
  distributions and local spectral structure, switching between an
  empirical-Bayes branch and a masked-baseline branch on bright bands.

Install from the repository root with `pip install -e .`.

```python
from racer_raman import ImageCalibrator, SpectrumCalibrator
```

## Workflow

1. Supply matching closed-shutter and open-shutter calibration acquisitions.
   The image path also requires a per-pixel instrumental background map
   (`instrument_background_adu`), fitted separately over an open-shutter
   exposure series; the calibration performs no exposure-series fit of its
   own and will not run without it.
2. Fit an image or spectrum calibrator.
3. Pass it one or more raw measurements.
4. Receive corrected output when an event is detected, or an unchanged copy when no correction is needed.

Calibration and measurement data must use the same exposure, detector readout, binning, gain, and region of interest.

## Examples

- [RACER-2D, image readout](notebooks/racer-2d.ipynb)
- [RACER-1D, batch row-summed spectra](notebooks/racer-1d.ipynb)

Measured closed-shutter and polystyrene Zarr data are included under [data](data/).

## License

MIT. See [LICENSE](LICENSE).

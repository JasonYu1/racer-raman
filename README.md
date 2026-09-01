# Cosmic Ray Bayes

Cosmic-ray detection and correction for detector images and row-summed spectra.

Install from the repository root with `pip install -e .`.

## Workflow

1. Supply matching closed-shutter and open-shutter calibration acquisitions.
2. Fit an image or spectrum calibrator.
3. Pass it a raw measurement.
4. Receive corrected output when an event is detected, or an unchanged copy when no correction is needed.

Calibration and measurement data must use the same exposure, detector readout, binning, gain, and region of interest.

## Examples

- [Image readout](notebooks/01_image_readout.ipynb)
- [Row-summed spectrum](notebooks/02_row_summed_readout.ipynb)

Measured closed-shutter and polystyrene Zarr data are included under [data](data/).

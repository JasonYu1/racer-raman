# Cosmic-ray correction for spectra

Fit a calibrator once, then pass it raw detector measurements. The calibrator
returns the corrected measurement on the same raw ADU scale.

## Install

```bash
python -m pip install ".[demos]"
```

## Image readout

```python
from cosmic_ray_bayes import ImageCalibrator, read_image_zarr

images = read_image_zarr(
    {
        "dark": "data/image/closed_shutter_2000ms.zarr",
        "polystyrene": "data/image/open_shutter_polystyrene_2000ms.zarr",
    },
    columns=range(1568, 1632),
)

dark = images.xs(("dark", 2000.0), level=("experiment", "exposure_ms"))
sample = images.xs(
    ("polystyrene", 2000.0), level=("experiment", "exposure_ms")
)
raw_image = sample.xs(67, level="frame")
training = sample.drop(index=67, level="frame")

fixed_image = ImageCalibrator.fit(dark, training).correct(raw_image)
```

`read_image_zarr` reads the exposure from each store and returns raw ADU in a
DataFrame indexed by `experiment`, `exposure_ms`, `frame`, and `row`. Its
detector columns retain their source coordinates. Row and column selections are
applied before the data are materialized, which avoids loading an entire large
image acquisition when only a detector region is needed.

## Row-summed spectrum

```python
from cosmic_ray_bayes import SpectrumCalibrator, read_spectrum_zarr

spectra = read_spectrum_zarr(
    {
        "dark": "data/spectrum/closed_shutter_1000ms_fvb.zarr",
        "polystyrene": "data/spectrum/open_shutter_polystyrene_1000ms_fvb.zarr",
    }
)

dark = spectra.xs(("dark", 1000.0), level=("experiment", "exposure_ms"))
sample = spectra.xs(
    ("polystyrene", 1000.0), level=("experiment", "exposure_ms")
)
raw_spectrum = sample.loc[42]
reference = sample.drop(index=42)

fixed_spectrum = SpectrumCalibrator.fit(
    dark,
    reference,
    exposure_ms=1000,
).correct(raw_spectrum)
```

`read_spectrum_zarr` returns raw ADU in a DataFrame indexed by `experiment`,
`exposure_ms`, and `frame`, with source detector coordinates as columns.

If the algorithm makes no correction, it returns an exact copy with the input
dtype and bytes preserved. If integer input needs a fractional correction, the
corrected result is `float64`; values outside the correction remain numerically
unchanged. Use `return_info=True` only when a correction mask or removed charge
is needed.

Calibration and measurement data must use the same exposure, detector readout,
binning, gain, and region of interest.

## Demonstrations

The notebooks fit on measured calibration acquisitions and correct separate,
held-out open-shutter measurements:

1. [`01_image_readout.ipynb`](notebooks/01_image_readout.ipynb) — corrects a
   measured polystyrene image and shows its row-summed 64-column detector segment
2. [`02_row_summed_readout.ipynb`](notebooks/02_row_summed_readout.ipynb) —
   corrects a measured, full-width row-summed polystyrene spectrum

The measured image and row-summed Zarr stores are under [`data/`](data/).

## Run the demonstrations

```bash
jupyter notebook notebooks
```

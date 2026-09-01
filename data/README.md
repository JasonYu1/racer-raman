# Demonstration data

All values are measured detector data. No peaks or particle events are
generated for the examples.

## Image Zarr stores

- `image/closed_shutter_2000ms.zarr` contains 1,000 measured dark frames.
- `image/open_shutter_polystyrene_2000ms.zarr` contains 100 measured
  open-shutter polystyrene frames. Frame 67 is the held-out measurement and has
  a natural event at detector row 16, column 1605.

Both stores contain raw `int32` ADU in `signal` with dimensions `(repeat,
detector_y, detector_x)`. They retain all 256 detector rows and global detector
columns 1568–1631. The coordinate arrays allow `read_image_zarr` to construct a
DataFrame indexed by experiment, exposure time, frame, and row.

## Row-summed Zarr stores

- `spectrum/closed_shutter_1000ms_fvb.zarr` contains 1,000 measured dark
  spectra.
- `spectrum/open_shutter_polystyrene_1000ms_fvb.zarr` contains 1,000 measured
  open-shutter polystyrene spectra. Frame 42 is the held-out measurement and has
  a natural event at detector columns 463–464.

Both stores contain raw `int32` ADU in `signal` with dimensions `(repeat,
detector_x)` and preserve global detector columns 0–1999. `read_spectrum_zarr`
constructs a DataFrame indexed by experiment, exposure time, and frame.

## Sanitization

The image stores are exact detector crops, and the row-summed stores are exact
copies of the corresponding source arrays. Acquisition timestamps, local paths,
stage coordinates, instrument-control identifiers, and unrelated metadata are
omitted.

These excerpts are included at the repository owner's direction. Add explicit
data-redistribution terms before changing the repository from private to public.

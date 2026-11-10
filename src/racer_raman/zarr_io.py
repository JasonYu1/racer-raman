"""Zarr ingestion for detector measurements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

AxisSelection = slice | Sequence[int] | np.ndarray | None
StorePath = str | PathLike[str]

__all__ = ["read_image_zarr", "read_spectrum_zarr"]


def _optional_dependencies() -> tuple[Any, Any]:
    try:
        import pandas as pd
        import zarr
    except ImportError as error:
        message = "Zarr ingestion requires pandas and zarr;"
        message += " install racer-raman[io]"
        raise ImportError(message) from error
    return pd, zarr


def _coordinate(root: Any, name: str, length: int) -> np.ndarray:
    if name not in root:
        return np.arange(length, dtype=np.int64)
    coordinate = np.asarray(root[name][:])
    if coordinate.ndim != 1 or coordinate.size != length:
        raise ValueError(f"Zarr coordinate {name!r} does not match the signal axis")
    if coordinate.dtype.kind not in "iu":
        raise ValueError(f"Zarr coordinate {name!r} must contain integers")
    if np.unique(coordinate).size != coordinate.size:
        raise ValueError(f"Zarr coordinate {name!r} must contain unique values")
    return coordinate.astype(np.int64, copy=False)


def _positions(
    coordinate: np.ndarray,
    selection: AxisSelection,
    axis_name: str,
) -> np.ndarray:
    if selection is None:
        result = np.arange(coordinate.size, dtype=np.intp)
    elif isinstance(selection, slice):
        step = 1 if selection.step is None else selection.step
        if step <= 0:
            raise ValueError(f"{axis_name} slice step must be positive")
        if np.any(np.diff(coordinate) <= 0):
            raise ValueError(
                f"{axis_name} coordinates must increase for slice selection"
            )
        start = (
            0
            if selection.start is None
            else int(np.searchsorted(coordinate, selection.start, side="left"))
        )
        stop = (
            coordinate.size
            if selection.stop is None
            else int(np.searchsorted(coordinate, selection.stop, side="left"))
        )
        result = np.arange(start, stop, step, dtype=np.intp)
    else:
        values = np.asarray(selection)
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError(f"{axis_name} selection must contain integer labels")
        lookup = {int(label): position for position, label in enumerate(coordinate)}
        missing = [int(value) for value in values if int(value) not in lookup]
        if missing:
            raise KeyError(
                f"{axis_name} labels are absent from the Zarr array: {missing}"
            )
        result = np.asarray([lookup[int(value)] for value in values], dtype=np.intp)
    if result.size == 0:
        raise ValueError(f"{axis_name} selection must not be empty")
    if np.unique(result).size != result.size:
        raise ValueError(f"{axis_name} selection must not contain duplicates")
    return result


def _frame_selections(
    stores: Mapping[str, StorePath],
    frames: Mapping[str, AxisSelection] | None,
) -> dict[str, AxisSelection]:
    if not isinstance(stores, Mapping) or not stores:
        raise ValueError("stores must map at least one experiment name to a Zarr path")
    experiments = tuple(stores)
    if any(not isinstance(name, str) or not name.strip() for name in experiments):
        raise ValueError("experiment names must be nonempty strings")
    if frames is not None and not isinstance(frames, Mapping):
        raise ValueError("frames must map experiment names to frame selections")
    selections = dict(frames or {})
    unknown = set(selections).difference(experiments)
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"frame selections reference unknown experiments: {names}")
    return selections


def _exposure_ms(root: Any, experiment: str) -> float:
    try:
        exposure_ms = float(root.attrs["exposure_ms"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Zarr experiment {experiment!r} requires a numeric exposure_ms attribute"
        ) from error
    if not np.isfinite(exposure_ms) or exposure_ms <= 0:
        raise ValueError(
            f"Zarr experiment {experiment!r} requires a positive exposure_ms"
        )
    return exposure_ms


def _open_signal(
    zarr: Any,
    experiment: str,
    store_path: StorePath,
    dimensions: tuple[str, ...],
    read_mode: str,
) -> tuple[Any, Any, float]:
    path = Path(store_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Zarr store does not exist: {path}")
    root = zarr.open_group(str(path), mode="r")
    variable_name = root.attrs.get("data_variable", "signal")
    if not isinstance(variable_name, str) or variable_name not in root:
        raise ValueError(
            f"Zarr experiment {experiment!r} does not contain its data variable"
        )
    signal = root[variable_name]
    if getattr(signal, "ndim", None) != len(dimensions):
        shape = ", ".join(dimensions)
        raise ValueError(
            f"Zarr experiment {experiment!r} must have dimensions ({shape})"
        )
    stored_dimensions = tuple(signal.attrs.get("_ARRAY_DIMENSIONS", ()))
    if stored_dimensions and stored_dimensions != dimensions:
        raise ValueError(
            f"Zarr experiment {experiment!r} has unsupported signal dimensions"
        )
    if np.dtype(signal.dtype).kind not in "uif":
        raise ValueError(
            f"Zarr experiment {experiment!r} must contain numeric detector values"
        )
    stored_read_mode = root.attrs.get("read_mode")
    if stored_read_mode not in (None, read_mode):
        raise ValueError(
            f"Zarr experiment {experiment!r} does not use {read_mode} readout"
        )
    return root, signal, _exposure_ms(root, experiment)


def _read_detector_zarr(
    stores: Mapping[str, StorePath],
    *,
    dimensions: tuple[str, ...],
    read_mode: str,
    frames: Mapping[str, AxisSelection] | None,
    rows: AxisSelection,
    columns: AxisSelection,
) -> pd.DataFrame:
    selections = _frame_selections(stores, frames)
    pd, zarr = _optional_dependencies()
    axis_names = {
        "repeat": "frame",
        "detector_y": "row",
        "detector_x": "column",
    }
    tables = []
    expected_geometry: tuple[np.ndarray, ...] | None = None

    for experiment, store_path in stores.items():
        root, signal, exposure_ms = _open_signal(
            zarr,
            experiment,
            store_path,
            dimensions,
            read_mode,
        )
        coordinates = [
            _coordinate(root, dimension, length)
            for dimension, length in zip(dimensions, signal.shape, strict=True)
        ]
        axis_selections = [selections.get(experiment)]
        if "detector_y" in dimensions:
            axis_selections.append(rows)
        axis_selections.append(columns)
        positions = [
            _positions(coordinate, selection, axis_names[dimension])
            for coordinate, selection, dimension in zip(
                coordinates,
                axis_selections,
                dimensions,
                strict=True,
            )
        ]
        labels = [
            coordinate[position]
            for coordinate, position in zip(coordinates, positions, strict=True)
        ]
        geometry = tuple(labels[1:])
        if expected_geometry is None:
            expected_geometry = geometry
        elif any(
            not np.array_equal(actual, expected)
            for actual, expected in zip(geometry, expected_geometry, strict=True)
        ):
            raise ValueError("selected detector geometry must match across stores")

        values = np.asarray(signal.oindex[tuple(positions)])
        index_labels = labels[:-1]
        index = pd.MultiIndex.from_product(
            ([experiment], [exposure_ms], *index_labels),
            names=(
                "experiment",
                "exposure_ms",
                *(axis_names[name] for name in dimensions[:-1]),
            ),
        )
        row_count = int(np.prod([label.size for label in index_labels]))
        tables.append(
            pd.DataFrame(
                values.reshape(row_count, labels[-1].size),
                index=index,
                columns=pd.Index(labels[-1], name="column"),
                copy=False,
            )
        )

    combined = pd.concat(tables, axis=0, copy=False)
    if not combined.index.is_unique:
        raise ValueError("combined Zarr index labels must be unique")
    return combined


def read_image_zarr(
    stores: Mapping[str, StorePath],
    *,
    frames: Mapping[str, AxisSelection] | None = None,
    rows: AxisSelection = None,
    columns: AxisSelection = None,
) -> pd.DataFrame:
    """Read image stores into an experiment, exposure, frame, and row table."""

    return _read_detector_zarr(
        stores,
        dimensions=("repeat", "detector_y", "detector_x"),
        read_mode="image",
        frames=frames,
        rows=rows,
        columns=columns,
    )


def read_spectrum_zarr(
    stores: Mapping[str, StorePath],
    *,
    frames: Mapping[str, AxisSelection] | None = None,
    columns: AxisSelection = None,
) -> pd.DataFrame:
    """Read FVB stores into an experiment, exposure, and frame table."""

    return _read_detector_zarr(
        stores,
        dimensions=("repeat", "detector_x"),
        read_mode="fvb",
        frames=frames,
        rows=None,
        columns=columns,
    )

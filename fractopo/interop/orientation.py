"""Complete trace orientations by fitting Fisher models to fracture-plane poles."""

from __future__ import annotations

from typing import Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from fractopo.general import NULL_SET

_MAX_KAPPA = 1e6


def dip_direction_to_pole(dip: float, dip_dir: float) -> np.ndarray:
    """Convert geological dip angles to an upward-pointing plane normal (ENU)."""
    dip_rad, direction_rad = np.deg2rad([dip, dip_dir % 360])
    return np.array(
        [
            -np.sin(dip_rad) * np.sin(direction_rad),
            -np.sin(dip_rad) * np.cos(direction_rad),
            np.cos(dip_rad),
        ]
    )


def pole_to_orientation(pole: np.ndarray) -> tuple[float, float]:
    """Convert a plane normal to canonical dip and dip direction."""
    pole = np.asarray(pole, dtype=float)
    pole = pole / np.linalg.norm(pole)
    if pole[2] < 0 or (
        np.isclose(pole[2], 0) and pole[np.flatnonzero(np.abs(pole) > 1e-12)[0]] < 0
    ):
        pole = -pole
    dip = np.rad2deg(np.arctan2(np.hypot(pole[0], pole[1]), pole[2]))
    dip_dir = np.rad2deg(np.arctan2(-pole[0], -pole[1])) % 360
    if np.hypot(pole[0], pole[1]) < 1e-12 or np.isclose(dip, 90.0):
        dip_dir = 0.0
    return float(dip), float(dip_dir)


def _valid_pairs(
    gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(gdf)
    dip = np.full(count, np.nan)
    dip_dir = np.full(count, np.nan)
    for column, target in (("dip", dip), ("dip_dir", dip_dir)):
        if column in gdf:
            target[:] = pd.to_numeric(gdf[column], errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
    valid = (
        np.isfinite(dip)
        & (dip >= 0)
        & (dip <= 90)
        & np.isfinite(dip_dir)
        & (dip_dir >= 0)
        & (dip_dir <= 360)
    )
    dip_dir[dip_dir == 360] = 0
    return dip, dip_dir, valid


def _fit_fisher(poles: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit an isotropic Fisher model after axial sign alignment."""
    scatter = poles.T @ poles
    _, eigenvectors = np.linalg.eigh(scatter)
    axis = eigenvectors[:, -1]
    if axis[2] < 0 or (
        np.isclose(axis[2], 0) and axis[np.flatnonzero(np.abs(axis) > 1e-12)[0]] < 0
    ):
        axis *= -1
    aligned = poles.copy()
    aligned[(aligned @ axis) < 0] *= -1
    resultant = np.linalg.norm(aligned.sum(axis=0)) / len(aligned)
    resultant = float(np.clip(resultant, 0, 1 - 1e-12))
    if resultant < 1e-8:
        kappa = 0.0
    else:
        kappa = resultant * (3 - resultant**2) / (1 - resultant**2)
    mean = aligned.mean(axis=0)
    if np.linalg.norm(mean) < 1e-12:
        mean = axis
        kappa = 0.0
    return mean / np.linalg.norm(mean), min(kappa, _MAX_KAPPA)


def _sample_fisher(
    mean: np.ndarray, kappa: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample a 3-D Fisher vector using inverse-CDF polar sampling."""
    if kappa >= _MAX_KAPPA * 0.999:
        return mean.copy()
    if kappa < 1e-8:
        z = rng.uniform(-1.0, 1.0)
    else:
        u = rng.random()
        z = 1 + np.logaddexp(np.log(u), np.log1p(-u) - 2 * kappa) / kappa
    phi = rng.uniform(0, 2 * np.pi)
    radial = np.sqrt(max(0.0, 1 - z * z))
    local = np.array([radial * np.cos(phi), radial * np.sin(phi), z])
    helper = (
        np.array([0.0, 0.0, 1.0]) if abs(mean[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    )
    first = np.cross(helper, mean)
    first /= np.linalg.norm(first)
    second = np.cross(mean, first)
    return local[0] * first + local[1] * second + local[2] * mean


def _angular_std(kappa: float) -> float:
    """Return the approximate Fisher angular standard deviation in degrees."""
    return float(np.rad2deg(np.sqrt(2 / kappa))) if kappa else 180.0


def _model_diagnostics(
    poles: np.ndarray, mean: np.ndarray, kappa: float, source: str
) -> dict[str, Any]:
    dip, dip_dir = pole_to_orientation(mean)
    return {
        "measured_count": len(poles),
        "mean_dip": dip,
        "mean_dip_dir": dip_dir,
        "angular_std": _angular_std(kappa),
        "kappa": float(kappa),
        "source": source,
    }


def _validate_fallback(
    dip: Optional[float], dip_dir: Optional[float], angular_std: Optional[float]
) -> None:
    if not (
        dip is not None
        and np.isfinite(dip)
        and 0 <= dip <= 90
        and dip_dir is not None
        and np.isfinite(dip_dir)
        and 0 <= dip_dir <= 360
        and angular_std is not None
        and np.isfinite(angular_std)
        and 0 <= angular_std <= 180
    ):
        raise ValueError(
            "fallback dip, dip direction, or angular spread is outside its valid range"
        )


def _fallback_kappa(angular_std: Optional[float]) -> float:
    assert angular_std is not None
    return (
        _MAX_KAPPA
        if angular_std == 0
        else min(2 / np.deg2rad(angular_std) ** 2, _MAX_KAPPA)
    )


def _fit_models(
    poles: np.ndarray,
    set_labels: np.ndarray,
    set_names: tuple[str, ...],
    valid: np.ndarray,
    fillable: np.ndarray,
    fill_null_set: bool,
    fallback_dip: Optional[float],
    fallback_dip_dir: Optional[float],
    fallback_angular_std: Optional[float],
) -> tuple[dict, dict, np.ndarray, float, str]:
    models = {}
    for name in set_names:
        selected = poles[set_labels[valid] == name]
        if len(selected) >= 2:
            models[name] = _fit_fisher(selected)
    diagnostics = {
        name: _model_diagnostics(poles[set_labels[valid] == name], *model, "set")
        for name, model in models.items()
    }
    if len(poles) >= 2:
        global_mean, global_kappa = _fit_fisher(poles)
        global_source = "global"
    elif any(
        value is None
        for value in (fallback_dip, fallback_dip_dir, fallback_angular_std)
    ):
        if np.any(~valid & fillable):
            raise ValueError(
                "at least two complete pairs or all fallback parameters are required"
            )
        global_mean, global_kappa, global_source = (
            np.array([0.0, 1.0, 0.0]),
            0.0,
            "global",
        )
    else:
        _validate_fallback(fallback_dip, fallback_dip_dir, fallback_angular_std)
        global_mean = dip_direction_to_pole(fallback_dip, fallback_dip_dir)
        global_kappa = _fallback_kappa(fallback_angular_std)
        global_source = "fallback"
    diagnostics["global"] = _model_diagnostics(
        poles, global_mean, global_kappa, global_source
    )
    for name in set_names:
        diagnostics.setdefault(
            name,
            _model_diagnostics(
                poles[set_labels[valid] == name],
                global_mean,
                global_kappa,
                global_source,
            ),
        )
        diagnostics[name]["source"] = "set" if name in models else global_source
    if fill_null_set:
        diagnostics[NULL_SET] = _model_diagnostics(
            poles, global_mean, global_kappa, global_source
        )
    return models, diagnostics, global_mean, global_kappa, global_source


def _fit_models(
    poles,
    set_labels,
    set_names,
    valid,
    fillable,
    fill_null_set,
    fallback_dip,
    fallback_dip_dir,
    fallback_angular_std,
):
    models = {
        name: _fit_fisher(selected)
        for name in set_names
        if len(selected := poles[set_labels[valid] == name]) >= 2
    }
    diagnostics = {
        name: _model_diagnostics(poles[set_labels[valid] == name], *model, "set")
        for name, model in models.items()
    }
    if len(poles) >= 2:
        global_mean, global_kappa = _fit_fisher(poles)
        global_source = "global"
    elif any(
        value is None
        for value in (fallback_dip, fallback_dip_dir, fallback_angular_std)
    ):
        if np.any(~valid & fillable):
            raise ValueError(
                "at least two complete pairs or all fallback parameters are required"
            )
        global_mean, global_kappa, global_source = (
            np.array([0.0, 1.0, 0.0]),
            0.0,
            "global",
        )
    else:
        _validate_fallback(fallback_dip, fallback_dip_dir, fallback_angular_std)
        global_mean = dip_direction_to_pole(fallback_dip, fallback_dip_dir)
        global_kappa = _fallback_kappa(fallback_angular_std)
        global_source = "fallback"
    diagnostics["global"] = _model_diagnostics(
        poles, global_mean, global_kappa, global_source
    )
    for name in set_names:
        diagnostics.setdefault(
            name,
            _model_diagnostics(
                poles[set_labels[valid] == name],
                global_mean,
                global_kappa,
                global_source,
            ),
        )
        diagnostics[name]["source"] = "set" if name in models else global_source
    if fill_null_set:
        diagnostics[NULL_SET] = _model_diagnostics(
            poles, global_mean, global_kappa, global_source
        )
    return models, diagnostics, global_mean, global_kappa


def _fill_orientations(
    gdf,
    labels,
    valid,
    dip,
    dip_dir,
    fillable,
    models,
    global_mean,
    global_kappa,
    diagnostics,
    rng,
):
    for index, label in enumerate(labels):
        if valid[index]:
            pole = dip_direction_to_pole(dip[index], dip_dir[index])
        elif fillable[index]:
            pole = _sample_fisher(*models.get(label, (global_mean, global_kappa)), rng)
        else:
            continue
        if pole[2] < 0 or (
            np.isclose(pole[2], 0) and pole[np.flatnonzero(np.abs(pole) > 1e-12)[0]] < 0
        ):
            pole = -pole
        if not valid[index]:
            sampled_dip, sampled_dir = pole_to_orientation(pole)
            gdf.iloc[index, gdf.columns.get_loc("dip")] = sampled_dip
            gdf.iloc[index, gdf.columns.get_loc("dip_dir")] = sampled_dir
            diagnostics[label]["filled_count"] = (
                diagnostics[label].get("filled_count", 0) + 1
            )
        for column, value in zip(("pole_east", "pole_north", "pole_up"), pole):
            gdf.iloc[index, gdf.columns.get_loc(column)] = value
    return gdf, diagnostics


def complete_network_orientations(
    network,
    rng: Optional[np.random.Generator] = None,
    fill_null_set: bool = False,
    fallback_dip: Optional[float] = None,
    fallback_dip_dir: Optional[float] = None,
    fallback_angular_std: Optional[float] = None,
) -> tuple[gpd.GeoDataFrame, dict[str, dict[str, Any]]]:
    """Complete missing trace orientations, returning a copy and diagnostics.

    A pair is usable when ``dip`` is in [0, 90] and ``dip_dir`` in [0, 360]
    (degrees clockwise from north). The values are converted to ENU unit plane
    poles. Axial poles are aligned to the principal scatter axis; set and global
    models use an isotropic Fisher distribution. At least two complete pairs
    fit a model; smaller sets borrow the global model. If the whole dataset has
    fewer than two pairs, supply all three fallback parameters. Angular spread
    is an approximate one-sigma value in degrees, capped internally at a
    Fisher concentration of 1e6. Pass a Generator for reproducible draws.

    Generated dip, dip_dir, and pole_east/north/up columns describe the same
    plane. Existing incomplete pairs are replaced together. NULL_SET rows are
    left alone unless ``fill_null_set`` is true.
    """
    if not isinstance(network.trace_gdf, gpd.GeoDataFrame):
        raise TypeError("network.trace_gdf must be a GeoDataFrame")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not isinstance(fill_null_set, (bool, np.bool_)):
        raise TypeError("fill_null_set must be a boolean")
    gdf = network.trace_gdf.copy()
    count = len(gdf)
    labels = np.asarray(network.trace_data.azimuth_set_array)
    if len(labels) != count or any(
        not isinstance(label, str) or not label for label in labels
    ):
        raise ValueError("trace set labels and trace_gdf rows are misaligned")
    dip, dip_dir, valid = _valid_pairs(gdf)
    if "dip" not in gdf:
        gdf["dip"] = np.nan
    if "dip_dir" not in gdf:
        gdf["dip_dir"] = np.nan
    for column in ("pole_east", "pole_north", "pole_up"):
        if column not in gdf:
            gdf[column] = np.nan
    set_labels = np.asarray(labels, dtype=str)
    set_names = tuple(label for label in dict.fromkeys(set_labels) if label != NULL_SET)
    fillable = (set_labels != NULL_SET) | fill_null_set
    poles = np.array(
        [dip_direction_to_pole(d, dd) for d, dd in zip(dip[valid], dip_dir[valid])]
    )
    models, diagnostics, global_mean, global_kappa = _fit_models(
        poles,
        set_labels,
        set_names,
        valid,
        fillable,
        fill_null_set,
        fallback_dip,
        fallback_dip_dir,
        fallback_angular_std,
    )
    generator = np.random.default_rng() if rng is None else rng
    return _fill_orientations(
        gdf,
        set_labels,
        valid,
        dip,
        dip_dir,
        fillable,
        models,
        global_mean,
        global_kappa,
        diagnostics,
        generator,
    )

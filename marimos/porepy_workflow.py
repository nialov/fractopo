# /// script
# [tool.marimo]
# version = "0.15.2"
# dependencies = ["marimo", "fractopo", "geopandas", "matplotlib", "numpy", "porepy", "mplstereonet"]
# [tool.marimo.runtime]
# on_cell_change = "autorun"
#
# [tool.ruff]
# extend-select = ["I"]
# ///

"""Demonstrate a reproducible fractopo-to-PorePy 3-D workflow."""

# Marimo cells intentionally import their own dependencies.
# ruff: noqa: PLC0415

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import geopandas as gpd
    import marimo as mo
    import matplotlib.pyplot as plt
    import mplstereonet
    import numpy as np

    from fractopo.analysis.network import Network
    from fractopo.interop.porepy import export_network_to_porepy_3d_csv_format

    return (
        Network,
        Path,
        export_network_to_porepy_3d_csv_format,
        gpd,
        mo,
        mplstereonet,
        np,
        plt,
    )


@app.cell
def _(mo):
    fracture_count = mo.ui.number(start=10, stop=500, value=60, step=10)
    seed = mo.ui.number(start=0, stop=999999, value=42, step=1)
    run = mo.ui.run_button(label="Generate PorePy network")
    mo.vstack(
        [
            mo.md("## Fractopo → PorePy 3-D fracture workflow"),
            mo.md(
                "This example uses the small **KB7** trace dataset. The bundled "
                "trace datasets do not include measured `dip`/`dip_dir` columns, "
                "so we create clearly labeled example orientation measurements: "
                "four traces per detected azimuth set. The workflow then fits "
                "set pole models, fills the remaining orientations, and exports."
            ),
            mo.hstack([mo.md("Fractures:"), fracture_count]),
            mo.hstack([mo.md("Random seed:"), seed]),
            run,
        ]
    )
    return fracture_count, run, seed


@app.cell
def _(Path, gpd):
    # Resolve from the repository root when run with `marimo edit` or `marimo run`.
    root = Path.cwd()
    if not (root / "tests/sample_data").exists():
        root = Path(__file__).resolve().parents[1]
    trace_path = root / "tests/sample_data/KB7/KB7_traces.geojson"
    area_path = root / "tests/sample_data/KB7/KB7_tulkinta_alue.geojson"
    traces = gpd.read_file(trace_path)
    area = gpd.read_file(area_path)
    return area, root, traces


@app.cell
def _(Network, area, np, traces):
    network = Network(
        trace_gdf=traces,
        area_gdf=area,
        name="KB7",
        determine_branches_nodes=False,
        truncate_traces=True,
    )
    # No bundled trace dataset has measured dip/dip_dir fields. Seed four
    # explicitly synthetic paired measurements per detected azimuth set.
    orientation_traces = network.trace_gdf.copy()
    orientation_traces["dip"] = np.nan
    orientation_traces["dip_dir"] = np.nan
    set_labels = network.trace_data.azimuth_set_array
    trace_azimuths = network.trace_data.azimuth_array
    measured_indices = np.concatenate(
        [
            np.flatnonzero(set_labels == name)[:4]
            for name in network.azimuth_set_names
            if np.count_nonzero(set_labels == name) >= 2
        ]
    )
    measured_count = len(measured_indices)
    orientation_traces.iloc[
        measured_indices, orientation_traces.columns.get_loc("dip")
    ] = np.tile([55.0, 62.0, 70.0, 78.0], measured_count // 4)
    orientation_traces.iloc[
        measured_indices, orientation_traces.columns.get_loc("dip_dir")
    ] = (trace_azimuths[measured_indices] + 90.0) % 360.0
    network.trace_gdf = orientation_traces
    orientation_diagnostics = network.complete_trace_orientations(
        rng=np.random.default_rng(42),
        fill_null_set=True,
        fallback_dip=75.0,
        fallback_dip_dir=90.0,
        fallback_angular_std=15.0,
    )
    network_summary = {
        "source traces": len(traces),
        "usable traces": len(network.trace_data.azimuth_array),
        "area": float(network.total_area),
        "azimuth sets": network.azimuth_set_names,
        "trace counts by set": network.trace_azimuth_set_counts,
        "mean trace length": float(np.mean(network.trace_data.length_array)),
        "measured orientation rows": measured_count,
        "generated orientation rows": int(
            network.trace_gdf[["dip", "dip_dir"]].notna().all(axis=1).sum()
            - measured_count
        ),
        "orientation models": orientation_diagnostics,
    }
    return network, network_summary


@app.cell
def _(mo, network_summary):
    mo.md(
        "### Network and orientation models\n\n"
        + "\n".join(f"- **{key}:** `{value}`" for key, value in network_summary.items())
    )


@app.cell
def _(
    area,
    export_network_to_porepy_3d_csv_format,
    fracture_count,
    mo,
    network,
    np,
    root,
    run,
    seed,
):
    mo.stop(not (run.value or mo.app_meta().mode == "script"))

    xmin, ymin, xmax, ymax = map(float, area.total_bounds)
    bounds = (xmin, ymin, 0.0, xmax, ymax, 20.0)
    csv_text = export_network_to_porepy_3d_csv_format(
        network,
        fracture_count=int(fracture_count.value),
        bounds=bounds,
        rng=np.random.default_rng(int(seed.value)),
    )
    output_path = root / ".cache" / "kb7_porepy_fractures.csv"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(csv_text + "\n")
    return csv_text, output_path


@app.cell
def _(csv_text, mo, network, output_path):
    rows = csv_text.splitlines()
    mo.md(
        "### Completed trace orientations and PorePy input\n\n"
        f"Completed orientations: **{network.trace_gdf[['dip', 'dip_dir']].notna().all(axis=1).sum()}**; "
        f"wrote `{output_path}` with **{len(rows) - 1}** circular fractures. "
        "The first row is the cuboid domain; each remaining row stores center, "
        "two semi-axes, strike, and dip."
    )


@app.cell
def _(mo, np, output_path):
    import porepy as pp

    porepy_network = pp.fracture_importer.network_from_csv(output_path, has_domain=True)
    fracture_centers = np.hstack(
        [np.asarray(fracture.center) for fracture in porepy_network.fractures]
    )
    mo.md(
        "### PorePy import\n\n"
        f"PorePy created a `{type(porepy_network).__name__}` containing "
        f"**{len(porepy_network.fractures)}** fractures. "
        f"Center array shape: `{fracture_centers.shape}`."
    )
    return fracture_centers, porepy_network


@app.cell
def _(
    area,
    fracture_centers,
    mo,
    mplstereonet,
    np,
    network,
    porepy_network,
    plt,
    traces,
):
    stereo_fig, stereo_axes = mplstereonet.subplots(1, 2, figsize=(10, 5))
    dip = network.trace_gdf["dip"].to_numpy(dtype=float)
    strike = (network.trace_gdf["dip_dir"].to_numpy(dtype=float) - 90.0) % 360.0
    measured = (
        network.trace_gdf["dip"].notna().to_numpy()
        & network.trace_gdf["dip_dir"].notna().to_numpy()
    )
    stereo_axes[0].pole(
        strike[measured], dip[measured], "o", markersize=2, color="0.55"
    )
    stereo_axes[0].density_contourf(
        strike[measured], dip[measured], measurement="poles", cmap="Greys", alpha=0.45
    )
    stereo_axes[0].set_title("Completed trace plane poles")
    exported_fractures = porepy_network.fractures
    exported_strikes = np.rad2deg(
        [fracture.strike_angle for fracture in exported_fractures]
    )
    exported_dips = np.rad2deg([fracture.dip_angle for fracture in exported_fractures])
    stereo_axes[1].pole(
        exported_strikes,
        exported_dips,
        "o",
        markersize=4,
        color="tab:blue",
    )
    if len(exported_fractures) > 1:
        stereo_axes[1].density_contourf(
            exported_strikes,
            exported_dips,
            measurement="poles",
            cmap="Blues",
            alpha=0.45,
        )
    stereo_axes[1].set_title("PorePy plane poles")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    traces.plot(ax=axes[0], color="0.55", linewidth=0.5)
    area.boundary.plot(ax=axes[0], color="black")
    axes[0].set_title("Input KB7 traces")
    axes[0].set_aspect("equal")
    axes[1].scatter(
        fracture_centers[0], fracture_centers[1], s=12, c=fracture_centers[2]
    )
    axes[1].set_title("Sampled PorePy fracture centers")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    mo.vstack(
        [
            stereo_fig,
            mo.md(
                "Stereonets show lower-hemisphere poles: the left net is the completed trace orientations; the right net is the PorePy-imported fracture orientations."
            ),
            fig,
            mo.md(
                "Color in the map panel represents sampled fracture-center z coordinates."
            ),
        ]
    )


if __name__ == "__main__":
    app.run()

# /// script
# [tool.marimo]
# version = "0.15.2"
# dependencies = ["marimo", "fractopo", "geopandas", "matplotlib", "numpy", "porepy"]
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

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import geopandas as gpd
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from fractopo.analysis.network import Network
    from fractopo.interop.porepy import export_network_to_porepy_3d_csv_format

    return (
        Network,
        Path,
        export_network_to_porepy_3d_csv_format,
        gpd,
        mo,
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
                "This example uses the small **KB7** trace dataset. "
                "The source traces are 2-D. This workflow detects 2-D azimuth "
                "sets, fits 3-D plane-pole models from a few example measurements, "
                "and fills the remaining orientations before PorePy export."
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
    return area, area_path, root, trace_path, traces


@app.cell
def _(Network, area, np, traces):
    # KB7 contains no orientation columns. Seed a small paired structural
    # measurement sample to demonstrate fitting from sparse measured data.
    traces_with_dips = traces.copy()
    traces_with_dips["dip"] = np.nan
    traces_with_dips["dip_dir"] = np.nan
    measured_count = min(12, len(traces_with_dips))
    measured_indices = traces_with_dips.index[:measured_count]
    traces_with_dips.loc[measured_indices, "dip"] = np.linspace(
        60.0, 85.0, measured_count
    )
    traces_with_dips.loc[measured_indices, "dip_dir"] = np.linspace(
        30.0, 150.0, measured_count
    )
    network = Network(
        trace_gdf=traces_with_dips,
        area_gdf=area,
        name="KB7",
        determine_branches_nodes=False,
        truncate_traces=True,
    )
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
    return network, network_summary, traces_with_dips


@app.cell
def _(mo, network_summary):
    mo.md(
        "### Network characterization\n\n"
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
    should_run = run.value or mo.app_meta().mode == "script"
    mo.stop(not should_run)

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
    return bounds, csv_text, output_path


@app.cell
def _(csv_text, mo, output_path):
    rows = csv_text.splitlines()
    mo.md(
        "### Generated PorePy input\n\n"
        f"Wrote `{output_path}` with **{len(rows) - 1}** circular fractures. "
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
    return fracture_centers, porepy_network, pp


@app.cell
def _(area, fracture_centers, mo, plt, traces):
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
        [fig, mo.md("Color in the right panel represents the sampled z coordinate.")]
    )


if __name__ == "__main__":
    app.run()

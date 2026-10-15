# /// script
# [tool.marimo]
# version = "0.15.2"
# dependencies = ["marimo", "fractopo", "geopandas", "matplotlib", "numpy", "porepy", "mplstereonet", "plotly"]
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
    import plotly.graph_objects as go
    import porepy as pp

    from fractopo.analysis.network import Network
    from fractopo.interop.porepy import (
        condition_porepy_csv_slice_targets,
        export_network_to_porepy_3d_csv_format,
        exterior_triangle_connectivity,
        porepy_csv_fracture_diameters,
        summarize_empirical_fracture_lengths,
        summarize_porepy_mesh,
        triangle_cell_connectivity,
    )

    return (
        Network,
        Path,
        condition_porepy_csv_slice_targets,
        export_network_to_porepy_3d_csv_format,
        exterior_triangle_connectivity,
        porepy_csv_fracture_diameters,
        summarize_empirical_fracture_lengths,
        summarize_porepy_mesh,
        triangle_cell_connectivity,
        gpd,
        mo,
        go,
        mplstereonet,
        np,
        plt,
        pp,
    )


@app.cell
def _(
    exterior_triangle_connectivity,
    np,
    pp,
    summarize_porepy_mesh,
    triangle_cell_connectivity,
):
    def load_or_mesh_network(
        porepy_network, mesh_path, reuse_existing, fracture_size, background_size
    ):
        if reuse_existing and mesh_path.is_file():
            mesh = pp.meshing.subdomains_to_mdg(
                pp.fracs.simplex.tetrahedral_grid_from_gmsh(mesh_path, verbose=0)
            )
            return mesh, "reused existing Gmsh mesh"

        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh = porepy_network.mesh(
            {
                "mesh_size_fracture": fracture_size,
                "mesh_size_boundary": background_size,
                "mesh_size_min": 0.1 * min(fracture_size, background_size),
                "refinement_size_multiplier": 1.0,
                "refinement_proximity_multiplier": 1.0,
            },
            file_name=mesh_path,
        )
        return mesh, "generated new Gmsh mesh"

    def mesh_arrays(mesh):
        volume_grid = mesh.subdomains(dim=3)[0]
        boundary_triangles = exterior_triangle_connectivity(
            volume_grid.face_nodes, volume_grid.cell_faces
        )
        coordinates, triangles, node_offset = [], [], 0
        for fracture_grid in mesh.subdomains(dim=2):
            fracture_coordinates = fracture_grid.nodes.T
            fracture_triangles = (
                triangle_cell_connectivity(fracture_grid.cell_nodes()) + node_offset
            )
            coordinates.append(fracture_coordinates)
            triangles.append(fracture_triangles)
            node_offset += len(fracture_coordinates)
        return (
            volume_grid.nodes,
            boundary_triangles,
            np.vstack(coordinates),
            np.vstack(triangles),
        )

    return load_or_mesh_network, mesh_arrays


@app.cell
def _(mo):
    fracture_count = mo.ui.number(start=10, stop=500, value=60, step=10)
    condition_on_slices = mo.ui.checkbox(
        label="Condition on surface slices", value=False
    )
    p21_tolerance = mo.ui.number(start=0.0, stop=1.0, value=0.5, step=0.05)
    top_count_tolerance = mo.ui.number(start=0.0, stop=1.0, value=0.5, step=0.05)
    seed = mo.ui.number(start=0, stop=999999, value=42, step=1)
    fracture_mesh_size = mo.ui.number(start=0.1, stop=10.0, value=3.0, step=0.1)
    background_mesh_size = mo.ui.number(start=0.1, stop=10.0, value=5.0, step=0.1)
    reuse_existing = mo.ui.checkbox(label="Reuse existing exports/meshes", value=True)
    run = mo.ui.run_button(label="Generate PorePy network")
    mesh_run = mo.ui.run_button(label="Mesh fracture network")
    mo.vstack(
        [
            mo.md("# Fractopo → PorePy 3-D fracture workflow"),
            mo.md("## Run settings"),
            mo.md(
                "This example uses the small **KB7** trace dataset. The bundled "
                "trace datasets do not include measured `dip`/`dip_dir` columns, "
                "so we create clearly labeled example orientation measurements: "
                "four traces per detected azimuth set. The workflow then fits "
                "set pole models, fills the remaining orientations, and exports."
            ),
            mo.hstack([mo.md("Base fractures:"), fracture_count]),
            condition_on_slices,
            mo.hstack([mo.md("P21 relative tolerance:"), p21_tolerance]),
            mo.hstack(
                [mo.md("Top trace-count relative tolerance:"), top_count_tolerance]
            ),
            mo.hstack([mo.md("Random seed:"), seed]),
            reuse_existing,
            run,
            mo.md("## Mesh settings (metres)"),
            mo.hstack([mo.md("Fracture size:"), fracture_mesh_size]),
            mo.hstack([mo.md("Background size:"), background_mesh_size]),
            mesh_run,
        ]
    )
    return (
        background_mesh_size,
        condition_on_slices,
        fracture_count,
        fracture_mesh_size,
        p21_tolerance,
        top_count_tolerance,
        mesh_run,
        reuse_existing,
        run,
        seed,
    )


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
        "## 1. Input data and orientation completion\n\n"
        + "\n".join(f"- **{key}:** `{value}`" for key, value in network_summary.items())
    )


@app.cell
def _(
    area,
    condition_on_slices,
    condition_porepy_csv_slice_targets,
    export_network_to_porepy_3d_csv_format,
    fracture_count,
    mo,
    p21_tolerance,
    top_count_tolerance,
    network,
    np,
    root,
    reuse_existing,
    run,
    seed,
):
    output_path = root / ".cache" / "kb7_porepy_fractures.csv"
    reuse_export = reuse_existing.value and output_path.is_file()
    generate_export = run.value or mo.app_meta().mode == "script"
    mo.stop(not (reuse_export or generate_export))

    slice_result = None
    if reuse_export:
        csv_text = output_path.read_text().rstrip("\n")
        export_source = "reused existing CSV"
    else:
        xmin, ymin, xmax, ymax = map(float, area.total_bounds)
        bounds = (xmin, ymin, 0.0, xmax, ymax, 20.0)
        rng = np.random.default_rng(int(seed.value))
        if condition_on_slices.value:
            section_heights = np.linspace(2.0, 18.0, 5)
            footprint_area = (xmax - xmin) * (ymax - ymin)
            slice_result = condition_porepy_csv_slice_targets(
                network,
                fracture_count=int(fracture_count.value),
                bounds=bounds,
                section_heights=section_heights,
                target_p21=float(network.trace_data.length_array.sum())
                / float(network.total_area),
                target_top_count=round(
                    len(network.trace_data.length_array)
                    * footprint_area
                    / float(network.total_area)
                ),
                rng=rng,
                max_attempts=3,
                p21_tolerance=float(p21_tolerance.value),
                top_count_tolerance=float(top_count_tolerance.value),
            )
            csv_text = slice_result["csv_text"]
        else:
            csv_text = export_network_to_porepy_3d_csv_format(
                network,
                fracture_count=int(fracture_count.value),
                bounds=bounds,
                rng=rng,
            )
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(csv_text + "\n")
        export_source = "generated conditioned CSV" if slice_result else "generated CSV"
    return csv_text, export_source, output_path, slice_result


@app.cell
def _(csv_text, export_source, mo, network, output_path, slice_result):
    rows = csv_text.splitlines()
    status = ""
    if slice_result:
        metrics = slice_result["metrics"]
        status = (
            f" Conditioning added **{slice_result['added_top_fracture_count']}** top-hit "
            f"fractures to the base population; top hits **{metrics['top_intersection_count']}**, "
            f"mean section P21 relative error **{metrics['mean_absolute_relative_p21_error']:.1%}** "
            f"in {slice_result['attempts']} attempt(s); both tolerances met: "
            f"**{slice_result['target_match']}**."
        )
    mo.md(
        "## 2. Generate fracture CSV\n\n"
        f"Completed orientations: **{network.trace_gdf[['dip', 'dip_dir']].notna().all(axis=1).sum()}**; "
        f"{export_source} at `{output_path}` with **{len(rows) - 1}** circular fractures. "
        "The first row is the cuboid domain; each remaining row stores center, "
        "two semi-axes, strike, and dip." + status
    )


@app.cell
def _(mo, np, output_path, pp):
    porepy_network = pp.fracture_importer.network_from_csv(output_path, has_domain=True)
    fracture_centers = np.hstack(
        [np.asarray(fracture.center) for fracture in porepy_network.fractures]
    )
    mo.md(
        "## 3. Validate PorePy import\n\n"
        f"PorePy created a `{type(porepy_network).__name__}` containing "
        f"**{len(porepy_network.fractures)}** fractures. "
        f"Center array shape: `{fracture_centers.shape}`."
    )
    return fracture_centers, porepy_network


@app.cell
def _(
    csv_text,
    mo,
    network,
    summarize_empirical_fracture_lengths,
):
    length_summary = summarize_empirical_fracture_lengths(network, csv_text)
    per_set = length_summary["by_set"]
    summary_lines = [
        f"- **{name}:** {values['count']} eligible traces; boundary-weighted "
        f"length min/median/mean/max `"
        f"{values['min']:.3g}/{values['median']:.3g}/{values['mean']:.3g}/"
        f"{values['max']:.3g}`."
        for name, values in per_set.items()
    ]
    generated = length_summary["generated"]
    summary_lines.append(
        f"- **Generated/reused CSV overall:** {generated['count']} fracture "
        f"diameters min/median/mean/max `"
        f"{generated['min']:.3g}/{generated['median']:.3g}/{generated['mean']:.3g}/"
        f"{generated['max']:.3g}`."
    )
    mo.md(
        "## 4. Inspect sampled fracture sizes\n\n"
        "Each eligible trace length in a set is sampled with equal probability; "
        "duplicate lengths therefore retain their observed frequency. The exporter "
        "uses these same boundary-weighted lengths as generated fracture diameters. "
        "The PorePy CSV contains no azimuth-set labels, so generated lengths are "
        "summarized overall rather than attributed back to sets.\n\n"
        + "\n".join(summary_lines)
    )


@app.cell
def _(
    background_mesh_size,
    load_or_mesh_network,
    mesh_run,
    mesh_arrays,
    mo,
    np,
    root,
    porepy_network,
    fracture_mesh_size,
    reuse_existing,
    summarize_porepy_mesh,
):
    mesh_directory = root / ".cache" / "porepy_mesh"
    mesh_path = mesh_directory / "kb7_fractures.msh"
    reuse_mesh = reuse_existing.value and mesh_path.is_file()
    mo.stop(not (reuse_mesh or mesh_run.value))

    mesh, mesh_source = load_or_mesh_network(
        porepy_network,
        mesh_path,
        reuse_mesh,
        float(fracture_mesh_size.value),
        float(background_mesh_size.value),
    )
    grids = mesh.subdomains()
    mesh_summary = {
        "mesh source": mesh_source,
        "mesh file": mesh_path,
        "fracture count": len(porepy_network.fractures),
        **summarize_porepy_mesh(
            np.asarray([grid.dim for grid in grids]),
            np.asarray([grid.num_cells for grid in grids]),
            len(mesh.interfaces()),
        ),
        "mesh artifacts": sorted(
            path.name for path in mesh_directory.glob("kb7_fractures.*")
        ),
    }
    mo.md(
        "## 5. Generate and inspect the mesh\n\n"
        + "\n".join(f"- **{key}:** `{value}`" for key, value in mesh_summary.items())
        + "\n\nAll Gmsh/PorePy output is written under `.cache/porepy_mesh/` "
        "by passing an explicit `file_name`; no mesh artifacts are expected in the working directory."
    )
    return mesh, mesh_path, mesh_summary


@app.cell
def _(go, mesh, mesh_arrays, mo):
    volume_nodes, boundary_triangles, coordinates, triangles = mesh_arrays(mesh)
    mesh_figure = go.Figure(
        data=[
            go.Mesh3d(
                x=volume_nodes[0],
                y=volume_nodes[1],
                z=volume_nodes[2],
                i=boundary_triangles[:, 0],
                j=boundary_triangles[:, 1],
                k=boundary_triangles[:, 2],
                color="lightsteelblue",
                opacity=0.2,
                name="domain boundary",
                hoverinfo="skip",
            ),
            go.Mesh3d(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                z=coordinates[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                color="crimson",
                opacity=0.9,
                name="fracture surfaces",
                hoverinfo="skip",
            ),
        ]
    )
    mesh_figure.update_layout(
        title="PorePy tetrahedral domain and fracture surfaces",
        scene={
            "aspectmode": "data",
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    mo.vstack(
        [
            mo.md("### 3-D mesh view"),
            mo.md(
                "Rotate or zoom the 3D view to inspect fracture placement in the domain."
            ),
            mesh_figure,
        ]
    )


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
            mo.md("## 6. Inspect orientations and spatial placement"),
            mo.md("### Plane-orientation stereonets"),
            stereo_fig,
            mo.md(
                "Stereonets show lower-hemisphere poles: the left net is the completed trace orientations; the right net is the PorePy-imported fracture orientations."
            ),
            mo.md("### Trace map and fracture-center locations"),
            fig,
            mo.md(
                "Color in the map panel represents sampled fracture-center z coordinates."
            ),
        ]
    )


if __name__ == "__main__":
    app.run()

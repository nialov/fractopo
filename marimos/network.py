# /// script
# [tool.marimo.runtime]
# on_cell_change = "autorun"
# ///

import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    import tempfile
    import zipfile
    from functools import partial
    from io import BytesIO
    from pathlib import Path

    import geopandas as gpd
    import marimo as mo
    import pyogrio
    import utils

    import fractopo.general
    import fractopo.tval.trace_validation
    from fractopo.analysis.network import Network
    return (
        BytesIO,
        Network,
        Path,
        fractopo,
        gpd,
        logging,
        mo,
        partial,
        pyogrio,
        tempfile,
        utils,
        zipfile,
    )


@app.cell
def _(mo):
    input_traces_file = mo.ui.file(kind="area")
    input_area_file = mo.ui.file(kind="area")
    input_trace_layer_name = mo.ui.text()
    input_area_layer_name = mo.ui.text()
    input_snap_threshold = mo.ui.text(value="0.001")
    input_contour_grid_cell_size = mo.ui.text(value="")
    input_name = mo.ui.text(value="Network")
    input_circular_target_area = mo.ui.switch(False)
    input_determine_branches_nodes = mo.ui.switch(True)
    input_truncate_traces = mo.ui.switch(True)
    input_debug = mo.ui.switch(False)
    input_define_azimuth_sets = mo.ui.switch(False)
    default_fits_to_plot = ("power_law", "lognormal", "exponential")
    input_fits_to_plot = mo.ui.multiselect(
        options=default_fits_to_plot, value=default_fits_to_plot
    )
    input_button = mo.ui.run_button()
    return (
        default_fits_to_plot,
        input_area_file,
        input_area_layer_name,
        input_button,
        input_circular_target_area,
        input_contour_grid_cell_size,
        input_debug,
        input_define_azimuth_sets,
        input_determine_branches_nodes,
        input_fits_to_plot,
        input_name,
        input_snap_threshold,
        input_trace_layer_name,
        input_traces_file,
        input_truncate_traces,
    )


@app.cell
def _(
    input_area_file,
    input_area_layer_name,
    input_circular_target_area,
    input_contour_grid_cell_size,
    input_debug,
    input_determine_branches_nodes,
    input_fits_to_plot,
    input_name,
    input_snap_threshold,
    input_trace_layer_name,
    input_traces_file,
    input_truncate_traces,
    mo,
    utils,
):
    prompts = [
        *utils.make_data_upload_prompts(
            input_traces_file=input_traces_file,
            input_area_file=input_area_file,
            input_trace_layer_name=input_trace_layer_name,
            input_area_layer_name=input_area_layer_name,
        ),
        utils.make_snap_threshold_hstack(input_snap_threshold),
        mo.hstack(
            [
                mo.md(
                    "[Contour grid cell size]"
                    + "(https://nialov.github.io/fractopo/notebooks/fractopo_network_1.html#Contour-Grids)"
                    + " (optional):"
                ),
                input_contour_grid_cell_size,
                "{}".format(input_contour_grid_cell_size.value),
            ]
        ),
        mo.md(f"Name for analysis: {input_name}"),
        mo.md(
            "[Is the target area a circle?]"
            + "(https://nialov.github.io/fractopo/notebooks/fractopo_network_1.html#Network)"
            + f" {input_circular_target_area}"
        ),
        mo.md(f"Determine branches and nodes? {input_determine_branches_nodes}"),
        mo.md(
            "[Truncate traces to target area?]"
            + "(https://nialov.github.io/fractopo/notebooks/fractopo_network_1.html#Network)"
            + f" {input_truncate_traces}"
        ),
        mo.hstack(
            [
                mo.md(
                    "Which [length distribution fits]"
                    + "(https://nialov.github.io/fractopo/auto_examples/plot_length_distributions.html#sphx-glr-auto-examples-plot-length-distributions-py)"
                    + " to plot:"
                ),
                input_fits_to_plot,
                "{}".format(input_fits_to_plot.value),
            ]
        ),
        mo.md(utils.ENABLE_VERBOSE_BASE.format(input_debug)),
    ]

    mo.vstack(prompts)
    return (prompts,)


@app.cell
def _(input_define_azimuth_sets, mo):
    mo.md(f"Define azimuth sets? {input_define_azimuth_sets}")
    return


@app.cell
def _(mo):
    input_azimuth_set_range_count = mo.ui.number(start=1, stop=10, value=3)
    return (input_azimuth_set_range_count,)


@app.cell
def _(input_azimuth_set_range_count, mo, partial):
    make_set_value = partial(mo.ui.number, start=0, stop=180, step=1)
    input_azimuth_set_ranges = mo.ui.array(
        [
            mo.ui.array([make_set_value(), make_set_value()])
            for _ in range(input_azimuth_set_range_count.value)
        ]
    )
    return input_azimuth_set_ranges, make_set_value


@app.cell
def _(input_azimuth_set_ranges, mo):
    default_set_names = [
        "-".join(map(str, set_range)) for set_range in input_azimuth_set_ranges.value
    ]
    input_azimuth_set_names = mo.ui.array(
        [mo.ui.text(set_name) for set_name in default_set_names]
    )
    return default_set_names, input_azimuth_set_names


@app.cell
def _(
    input_azimuth_set_names,
    input_azimuth_set_range_count,
    input_azimuth_set_ranges,
    input_define_azimuth_sets,
    mo,
):
    input_azimuth_stack = mo.vstack(
        [
            mo.md(f"Number of sets: {input_azimuth_set_range_count}"),
            mo.md(f"Set ranges (from 0 to 180): {input_azimuth_set_ranges}"),
            mo.md(f"Set names: {input_azimuth_set_names}"),
        ]
    )
    if input_define_azimuth_sets.value:
        mo.output.replace(input_azimuth_stack)
    return (input_azimuth_stack,)


@app.cell
def _(input_button, mo):
    mo.md(f"Press to (re)start analysis: {input_button}")
    return


@app.cell
def _(
    fractopo,
    input_area_file,
    input_area_layer_name,
    input_azimuth_set_names,
    input_azimuth_set_ranges,
    input_button,
    input_circular_target_area,
    input_contour_grid_cell_size,
    input_define_azimuth_sets,
    input_determine_branches_nodes,
    input_fits_to_plot,
    input_snap_threshold,
    input_trace_layer_name,
    input_traces_file,
    input_truncate_traces,
    mo,
    utils,
):
    def execute():
        cli_args = mo.cli_args()
        if len(cli_args) != 0:
            (
                name,
                traces_gdf,
                area_gdf,
                snap_threshold,
                contour_grid_cell_size,
                azimuth_set_ranges,
                azimuth_set_names,
                fits_to_plot,
            ) = utils.parse_network_cli_args(cli_args)

        else:
            mo.stop(not input_button.value)
            (
                name,
                traces_gdf,
                area_gdf,
                snap_threshold,
                contour_grid_cell_size,
                azimuth_set_ranges,
                azimuth_set_names,
                fits_to_plot,
            ) = utils.parse_network_app_args(
                input_trace_layer_name,
                input_area_layer_name,
                input_traces_file,
                input_area_file,
                input_snap_threshold,
                input_contour_grid_cell_size,
                input_define_azimuth_sets,
                input_azimuth_set_ranges,
                input_azimuth_set_names,
                input_fits_to_plot,
            )

        network = fractopo.analysis.network.Network(
            trace_gdf=traces_gdf,
            area_gdf=area_gdf,
            name=name,
            circular_target_area=input_circular_target_area.value,
            determine_branches_nodes=input_determine_branches_nodes.value,
            truncate_traces=input_truncate_traces.value,
            snap_threshold=snap_threshold,
            azimuth_set_ranges=azimuth_set_ranges,
            azimuth_set_names=azimuth_set_names,
        )

        return network, name, contour_grid_cell_size, fits_to_plot
    return (execute,)


@app.cell
def _(execute, input_debug, mo, partial, utils):
    execute_results, execute_exception, execute_stderr_and_stdout = (
        utils.capture_function_outputs(execute)
    )
    report_output = partial(
        mo.output.replace,
        "\n".join([execute_stderr_and_stdout, str(execute_exception)]),
    )
    if input_debug.value or execute_exception is not None:
        report_output()
    if execute_exception is not None:
        network = None
        name = None
        contour_grid_cell_size = None
        fits_to_plot = None
    else:
        network, name, contour_grid_cell_size, fits_to_plot = execute_results
    return (
        contour_grid_cell_size,
        execute_exception,
        execute_results,
        execute_stderr_and_stdout,
        fits_to_plot,
        name,
        network,
        report_output,
    )


@app.cell
def _(mo):
    mo.md("""## Results""")
    return


@app.cell
def _(mo, network):
    if network is None:
        mo.output.replace("")
    else:
        mo.output.replace(network.parameters)
    return


@app.cell
def _(
    contour_grid_cell_size,
    fits_to_plot,
    input_determine_branches_nodes,
    mo,
    name,
    network,
    partial,
    utils,
):
    download_element, to_file_exception, to_file_stderr_and_stdout = (
        utils.capture_function_outputs(
            partial(
                utils.network_results_to_download_element,
                network=network,
                determine_branches_nodes=input_determine_branches_nodes.value,
                name=name,
                contour_grid_cell_size=contour_grid_cell_size,
                fits_to_plot=fits_to_plot,
            )
        )
    )
    if to_file_exception is not None:
        mo.output.replace(
            "\n".join([to_file_stderr_and_stdout, str(to_file_exception)])
        )
    return download_element, to_file_exception, to_file_stderr_and_stdout


@app.cell
def _(execute_exception, mo, to_file_exception):
    if len(mo.cli_args()) != 0:
        if execute_exception is not None:
            raise execute_exception
        if to_file_exception is not None:
            raise to_file_exception
    return


@app.cell
def _(download_element, mo):
    if download_element is not None:
        mo.output.replace(
            mo.md(f"### Download network analysis results: {download_element}")
        )
    else:
        mo.output.replace(
            mo.md("### Failed to analyze trace data. Nothing to download.")
        )
    return


if __name__ == "__main__":
    app.run()

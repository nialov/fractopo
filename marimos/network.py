# /// script
# [tool.marimo.runtime]
# on_cell_change = "autorun"
# ///

import marimo

__generated_with = "0.9.1"
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

    return (
        BytesIO,
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
    input_name = mo.ui.text(value="Network")
    input_circular_target_area = mo.ui.switch(False)
    input_determine_branches_nodes = mo.ui.switch(True)
    input_truncate_traces = mo.ui.switch(True)
    input_debug = mo.ui.switch(False)
    input_button = mo.ui.run_button()
    return (
        input_area_file,
        input_area_layer_name,
        input_button,
        input_circular_target_area,
        input_debug,
        input_determine_branches_nodes,
        input_name,
        input_snap_threshold,
        input_trace_layer_name,
        input_traces_file,
        input_truncate_traces,
    )


@app.cell
def __(
    input_area_file,
    input_area_layer_name,
    input_button,
    input_circular_target_area,
    input_debug,
    input_determine_branches_nodes,
    input_name,
    input_snap_threshold,
    input_trace_layer_name,
    input_traces_file,
    input_truncate_traces,
    mo,
):
    prompts = [
        mo.md(f"## Upload trace data: {input_traces_file}"),
        mo.md(f"Trace layer name, if applicable: {input_trace_layer_name}"),
        mo.md(f"## Upload area data: {input_area_file}"),
        mo.md(f"Area layer name, if applicable: {input_area_layer_name}"),
        mo.hstack(
            [
                "Snap threshold:",
                input_snap_threshold,
                "{}".format(input_snap_threshold.value),
            ]
        ),
        mo.md(f"Name for analysis: {input_name}"),
        mo.md(f"Is the target area a circle? {input_circular_target_area}"),
        mo.md(f"Determine branches and nodes? {input_determine_branches_nodes}"),
        mo.md(f"Truncate traces to target area? {input_truncate_traces}"),
        mo.md(f"Enable verbose debug output? {input_debug}"),
        mo.md(f"Press to (re)start analysis: {input_button}"),
    ]

    mo.vstack(prompts)
    return (prompts,)


@app.cell
def _(
    fractopo,
    input_area_file,
    input_area_layer_name,
    input_button,
    input_circular_target_area,
    input_determine_branches_nodes,
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
            name, traces_gdf, area_gdf, snap_threshold = utils.parse_network_cli_args(
                cli_args
            )
        else:
            mo.stop(not input_button.value)
            name, traces_gdf, area_gdf, snap_threshold = utils.parse_network_app_args(
                input_trace_layer_name,
                input_area_layer_name,
                input_traces_file,
                input_area_file,
                input_snap_threshold,
            )

        network = fractopo.analysis.network.Network(
            trace_gdf=traces_gdf,
            area_gdf=area_gdf,
            name=name,
            circular_target_area=input_circular_target_area.value,
            determine_branches_nodes=input_determine_branches_nodes.value,
            truncate_traces=input_truncate_traces.value,
            snap_threshold=snap_threshold,
        )

        return network, name

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
    else:
        network, name = execute_results
    return (
        execute_exception,
        execute_results,
        execute_stderr_and_stdout,
        name,
        network,
        report_output,
    )


@app.cell
def __(mo):
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
def __(input_determine_branches_nodes, mo, name, network, partial, utils):
    download_element, to_file_exception, to_file_stderr_and_stdout = (
        utils.capture_function_outputs(
            partial(
                utils.network_results_to_download_element,
                network=network,
                determine_branches_nodes=input_determine_branches_nodes.value,
                name=name,
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
def __(download_element, mo):
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

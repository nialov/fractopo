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
        utils,
    )


@app.cell
def _(mo):
    input_traces_file = mo.ui.file(kind="area")
    input_area_file = mo.ui.file(kind="area")
    input_trace_layer_name = mo.ui.text()
    input_area_layer_name = mo.ui.text()
    input_snap_threshold = mo.ui.text(value="0.001")
    input_debug = mo.ui.switch(False)
    input_button = mo.ui.run_button()
    return (
        input_area_file,
        input_area_layer_name,
        input_button,
        input_debug,
        input_snap_threshold,
        input_trace_layer_name,
        input_traces_file,
    )


@app.cell
def __(
    input_area_file,
    input_area_layer_name,
    input_button,
    input_debug,
    input_snap_threshold,
    input_trace_layer_name,
    input_traces_file,
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
        mo.md(utils.ENABLE_VERBOSE_BASE.format(input_debug)),
        mo.md(f"Press to (re)start validation: {input_button}"),
    ]

    mo.vstack(prompts)
    return (prompts,)


@app.cell
def _(
    fractopo,
    input_area_file,
    input_area_layer_name,
    input_button,
    input_snap_threshold,
    input_trace_layer_name,
    input_traces_file,
    mo,
    utils,
):
    def execute():
        cli_args = mo.cli_args()
        if len(cli_args) != 0:
            name, traces_gdf, area_gdf, snap_threshold = (
                utils.parse_validation_cli_args(cli_args)
            )
        else:
            mo.stop(not input_button.value)
            name, traces_gdf, area_gdf, snap_threshold = (
                utils.parse_validation_app_args(
                    input_trace_layer_name,
                    input_area_layer_name,
                    input_traces_file,
                    input_area_file,
                    input_snap_threshold,
                )
            )

        validation = fractopo.tval.trace_validation.Validation(
            traces=traces_gdf,
            area=area_gdf,
            name=name,
            allow_fix=True,
            SNAP_THRESHOLD=snap_threshold,
        )

        validated = validation.run_validation()

        validated_clean = fractopo.general.convert_list_columns(validated, allow=True)

        return validated_clean, name

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
        validated_clean = None
        name = None
    else:
        validated_clean, name = execute_results
    return (
        execute_exception,
        execute_results,
        execute_stderr_and_stdout,
        name,
        report_output,
        validated_clean,
    )


@app.cell
def __(mo):
    mo.md("""## Results""")
    return


@app.cell
def _(fractopo, mo, validated_clean):
    if validated_clean is None:
        mo.output.replace("")
    elif (
        fractopo.tval.trace_validation.Validation.ERROR_COLUMN
        in validated_clean.columns
    ):
        mo.output.replace(
            validated_clean[
                fractopo.tval.trace_validation.Validation.ERROR_COLUMN
            ].value_counts()
        )
    return


@app.cell
def _(mo, validated_clean):
    if validated_clean is not None:
        mo.output.replace(validated_clean.drop(columns=["geometry"]))
    return


@app.cell
def __(mo, name, partial, utils, validated_clean):
    download_element, to_file_exception, to_file_stderr_and_stdout = (
        utils.capture_function_outputs(
            partial(
                utils.validated_clean_to_download_element,
                validated_clean=validated_clean,
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
        mo.output.replace(mo.md(f"### Download validated data: {download_element}"))
    else:
        mo.output.replace(
            mo.md(
                "### Failed to validate or write validated to file. Nothing to download."
            )
        )
    return


if __name__ == "__main__":
    app.run()

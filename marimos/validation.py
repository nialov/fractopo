# ruff: noqa: PLC0415
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
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(
    fractopo,
    input_area_file,
    input_area_layer_name,
    input_button,
    input_snap_threshold,
    input_trace_layer_name,
    input_traces_file,
    is_script_mode,
    mo,
    utils,
):
    def execute():
        if not is_script_mode:
            mo.stop(not input_button.value)
        name, traces_gdf, area_gdf, snap_threshold = utils.parse_validation_app_args(
            input_trace_layer_name,
            input_area_layer_name,
            input_traces_file,
            input_area_file,
            input_snap_threshold,
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


@app.cell
def _(mo, validated_clean):
    if validated_clean is not None:
        mo.output.replace(validated_clean.drop(columns=["geometry"]))


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
def _():
    return


@app.cell
def _(fractopo, utils):
    import pytest

    @pytest.fixture()
    def traces_gdf(traces_area_name):
        import geopandas as gpd

        traces_path, _, _ = traces_area_name
        return gpd.read_file(traces_path)

    @pytest.fixture()
    def area_gdf(traces_area_name):
        import geopandas as gpd

        _, area_path, _ = traces_area_name
        return gpd.read_file(area_path)

    @pytest.fixture()
    def dataset_name(traces_area_name):
        _, _, name = traces_area_name
        return name

    def test_validation_runs(traces_gdf, area_gdf, dataset_name):
        """Integration test: run validation on sample data."""
        validation = fractopo.tval.trace_validation.Validation(
            traces=traces_gdf,
            area=area_gdf,
            name=dataset_name,
            allow_fix=True,
            SNAP_THRESHOLD=fractopo.tval.trace_validation.Validation.SNAP_THRESHOLD,
        )
        validated = validation.run_validation()
        validated_clean = fractopo.general.convert_list_columns(validated, allow=True)
        assert validated_clean is not None
        assert len(validated_clean) > 0

    def test_validation_download(traces_gdf, area_gdf, dataset_name):
        """Integration test: validate and produce download artifact."""
        validation = fractopo.tval.trace_validation.Validation(
            traces=traces_gdf,
            area=area_gdf,
            name=dataset_name,
            allow_fix=True,
            SNAP_THRESHOLD=fractopo.tval.trace_validation.Validation.SNAP_THRESHOLD,
        )
        validated = validation.run_validation()
        validated_clean = fractopo.general.convert_list_columns(validated, allow=True)
        download_element = utils.validated_clean_to_download_element(
            validated_clean=validated_clean, name=dataset_name
        )
        assert download_element is not None


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


if __name__ == "__main__":
    app.run()

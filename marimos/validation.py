import marimo

__generated_with = "0.9.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    from io import BytesIO
    from pathlib import Path

    import geopandas as gpd
    import marimo as mo
    import pyogrio

    import fractopo.general
    import fractopo.tval.trace_validation

    return BytesIO, Path, fractopo, gpd, logging, mo, pyogrio


@app.cell
def _(mo):
    input_traces_file = mo.ui.file(kind="area")
    input_area_file = mo.ui.file(kind="area")
    input_trace_layer_name = mo.ui.text()
    input_area_layer_name = mo.ui.text()
    input_snap_threshold = mo.ui.text(value="0.001")
    input_button = mo.ui.run_button()
    return (
        input_area_file,
        input_area_layer_name,
        input_button,
        input_snap_threshold,
        input_trace_layer_name,
        input_traces_file,
    )


@app.cell
def __(
    input_area_file,
    input_area_layer_name,
    input_button,
    input_snap_threshold,
    input_trace_layer_name,
    input_traces_file,
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
        mo.md(f"Press to (re)start validation: {input_button}"),
    ]

    mo.vstack(prompts)
    return (prompts,)


@app.cell
def _(
    Path,
    fractopo,
    gpd,
    input_area_file,
    input_area_layer_name,
    input_button,
    input_snap_threshold,
    input_trace_layer_name,
    input_traces_file,
    mo,
    pyogrio,
):
    def execute():
        cli_args = mo.cli_args()
        if len(cli_args) != 0:
            cli_traces_path = Path(cli_args.get("traces-path"))
            cli_area_path = Path(cli_args.get("area-path"))

            name = cli_args.get("name") or cli_traces_path.stem

            driver = pyogrio.detect_write_driver(cli_traces_path.name)

            traces_gdf = gpd.read_file(cli_traces_path, driver=driver)
            area_gdf = gpd.read_file(cli_area_path, driver=driver)
            snap_threshold_str = cli_args.get("snap-threshold")
            if snap_threshold_str is None:
                snap_threshold = (
                    fractopo.tval.trace_validation.Validation.SNAP_THRESHOLD
                )
            else:
                snap_threshold = float(snap_threshold_str)
        else:
            mo.stop(not input_button.value)

            trace_layer_name = (
                input_trace_layer_name.value
                if input_trace_layer_name.value != ""
                else None
            )
            area_layer_name = (
                input_area_layer_name.value
                if input_area_layer_name.value != ""
                else None
            )

            driver = pyogrio.detect_write_driver(input_traces_file.name())
            print(f"Detected driver: {driver}")

            print(
                f"Trace layer name: {trace_layer_name}"
                if trace_layer_name is not None
                else "No layer specified"
            )
            traces_gdf = gpd.read_file(
                input_traces_file.contents(),
                layer=trace_layer_name,
                # , driver=driver
            )
            print(
                f"Area layer name: {area_layer_name}"
                if area_layer_name is not None
                else "No layer specified"
            )
            area_gdf = gpd.read_file(
                input_area_file.contents(),
                layer=area_layer_name,
                # , driver=driver
            )

            snap_threshold = float(input_snap_threshold.value)
            print(f"Snap threshold: {snap_threshold}")

            name = (
                Path(input_traces_file.name()).stem
                if trace_layer_name is None
                else trace_layer_name
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
def _(execute, logging, mo):
    with mo.redirect_stderr():
        with mo.redirect_stdout():
            try:
                validated_clean, name = execute()
                execute_exception = None
            except Exception as exc:
                logging.error("Failed to validate input data.", exc_info=True)
                validated_clean = None
                name = None
                execute_exception = exc
    return execute_exception, name, validated_clean


@app.cell
def __(mo):
    mo.md("## Results")
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
def __(BytesIO, mo, name, validated_clean):
    def to_file():
        if validated_clean is not None:
            validated_clean_file = BytesIO()
            validated_clean.to_file(validated_clean_file, driver="GPKG", layer=name)
            validated_clean_file.seek(0)
            download_element = mo.download(
                data=validated_clean_file,
                filename=f"{name}_validated.gpkg",
                mimetype="application/octet-stream",
            )
        else:
            download_element = None
        return download_element

    return (to_file,)


@app.cell
def __(logging, mo, to_file):
    with mo.redirect_stderr():
        with mo.redirect_stdout():
            try:
                download_element = to_file()
                to_file_exception = None
            except Exception as exc:
                logging.error("Failed to write validated data.", exc_info=True)
                to_file_exception = exc
                download_element = None

    return download_element, to_file_exception


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

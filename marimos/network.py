import marimo

__generated_with = "0.9.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    import tempfile
    import zipfile
    from io import BytesIO
    from pathlib import Path

    import geopandas as gpd
    import marimo as mo
    import pyogrio

    import fractopo.general
    import fractopo.tval.trace_validation

    return (
        BytesIO,
        Path,
        fractopo,
        gpd,
        logging,
        mo,
        pyogrio,
        tempfile,
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
    input_button = mo.ui.run_button()
    return (
        input_area_file,
        input_area_layer_name,
        input_button,
        input_circular_target_area,
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
        mo.md(f"Press to (re)start analysis: {input_button}"),
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
    input_circular_target_area,
    input_determine_branches_nodes,
    input_snap_threshold,
    input_trace_layer_name,
    input_traces_file,
    input_truncate_traces,
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
            name = (
                Path(input_traces_file.name()).stem
                if trace_layer_name is None
                else trace_layer_name
            )

            print(
                str.join(
                    "\n",
                    [
                        f"Snap threshold: {snap_threshold}",
                        f"Name: {name}",
                    ],
                )
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
def _(execute, logging, mo):
    with mo.redirect_stderr():
        with mo.redirect_stdout():
            try:
                network, name = execute()
                execute_exception = None
            except Exception as exc:
                logging.error("Failed to analyze trace data.", exc_info=True)
                network = None
                name = None
                execute_exception = exc
    return execute_exception, name, network


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
def __(
    BytesIO,
    Path,
    input_determine_branches_nodes,
    mo,
    name,
    network,
    tempfile,
    zipfile,
):
    def to_file():
        if network is not None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)

                # Create and write plots to tmp_dir
                if input_determine_branches_nodes.value:
                    network.export_network_analysis(output_path=tmp_dir_path)
                else:
                    _, fig, _ = network.plot_trace_azimuth()
                    fig.savefig(tmp_dir_path / "trace_azimuth.png", bbox_inches="tight")
                    _, fig, _ = network.plot_trace_lengths()
                    fig.savefig(tmp_dir_path / "trace_lengths.png", bbox_inches="tight")

                zip_io = BytesIO()

                # Open an in-memory zip file
                with zipfile.ZipFile(zip_io, mode="a") as zip_file:
                    for path in tmp_dir_path.rglob("*"):
                        # Do not add directories, only files
                        if path.is_dir():
                            continue
                        path_rel = path.relative_to(tmp_dir_path)

                        # Write file in-memory to zip file
                        zip_file.write(path, arcname=path_rel)

            # Move to start of file
            zip_io.seek(0)

            download_element = mo.download(
                data=zip_io,
                filename=f"{name}.zip",
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
                logging.error("Failed to write results.", exc_info=True)
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

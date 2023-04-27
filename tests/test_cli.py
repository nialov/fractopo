"""
Tests for command line entrypoints.
"""
from pathlib import Path

import geopandas as gpd
import pytest
from typer.testing import CliRunner as TyperCliRunner

import tests
from fractopo import cli
from fractopo.tval.trace_validation import Validation

typer_cli_runner = TyperCliRunner()


@pytest.mark.parametrize(
    "trace_path, area_path, auto_fix", tests.test_tracevalidate_params
)
@pytest.mark.parametrize("snap_threshold", [0.01, 0.001])
def test_tracevalidate_typer(
    trace_path: Path,
    area_path: Path,
    auto_fix: str,
    tmp_path: Path,
    snap_threshold: float,
):
    """
    Tests tracevalidate typer functionality.
    """
    clirunner = TyperCliRunner()
    output_file = tmp_path / f"{trace_path.stem}.{trace_path.suffix}"
    cli_args = [
        "tracevalidate",
        str(trace_path),
        str(area_path),
        auto_fix,
        "--output",
        str(output_file),
        "--summary",
        "--snap-threshold",
        # should be valid for both 0.01 and 0.001
        str(snap_threshold),
    ]
    result = clirunner.invoke(cli.APP, cli_args)
    # Check that exit code is 0 (i.e. ran successfully.)
    tests.click_error_print(result)
    # Checks if output is saved
    assert output_file.exists()
    output_gdf = gpd.read_file(output_file)
    assert isinstance(output_gdf, gpd.GeoDataFrame)
    assert output_gdf.crs == gpd.read_file(trace_path).crs
    assert Validation.ERROR_COLUMN in output_gdf.columns
    if "--summary" in cli_args:
        assert "Out of" in result.output
        if not "There were" in result.output:
            assert "0 were invalid" in result.output


def test_make_output_dir(tmp_path):
    """
    Test make_output_dir.
    """
    some_dir = Path(tmp_path)
    output_dir = cli.make_output_dir(some_dir)
    assert output_dir.exists()
    assert output_dir.is_dir()


@pytest.mark.parametrize("args", tests.test_tracevalidate_only_area_params)
def test_tracevalidate_only_area(args, tmp_path):
    """
    Test tracevalidate script with --only-area-validation.
    """
    outputs_cmds = ["--output", str(tmp_path / "output_traces")]
    clirunner = TyperCliRunner()
    result = clirunner.invoke(cli.APP, ["tracevalidate"] + args + outputs_cmds)
    # Check that exit code is 0 (i.e. ran successfully.)
    tests.click_error_print(result)

    assert Path(outputs_cmds[1]).exists()
    assert Validation.ERROR_COLUMN in gpd.read_file(outputs_cmds[1]).columns


@pytest.mark.parametrize(
    "traces_path,area_path,determine_branches_nodes",
    [
        (tests.kb7_trace_50_path, tests.kb7_area_path, True),
        (tests.kb7_trace_50_path, tests.kb7_area_path, False),
    ],
)
def test_fractopo_network_cli(
    traces_path, area_path, determine_branches_nodes, tmp_path
):
    """
    Test fractopo network cli entrypoint.
    """
    tmp_path.mkdir(exist_ok=True)
    result = typer_cli_runner.invoke(
        cli.APP,
        [
            "network",
            str(traces_path),
            str(area_path),
            "--general-output",
            str(tmp_path),
        ]
        + ([] if determine_branches_nodes else ["--no-determine-branches-nodes"]),
    )

    tests.click_error_print(result)

    output_files = list(tmp_path.glob("*"))
    output_files_names = [path.name for path in output_files]
    assert len(output_files) > 0

    assert "branches" in str(output_files) or not determine_branches_nodes
    assert "nodes" in str(output_files) or not determine_branches_nodes

    assert len(list(tmp_path.glob("*.svg"))) > 0

    assert "additional_numerical_data.json" in output_files_names

    # Sanity check
    assert "adddditional_numerical_data.json" not in output_files_names


@pytest.mark.parametrize(
    "logging_level_str", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
)
def test_fractopo_callback(logging_level_str: str):
    """
    Test fractopo_callback.
    """
    result = typer_cli_runner.invoke(
        cli.APP,
        [
            "--log-level",
            logging_level_str,
            "info",
        ],
    )
    tests.click_error_print(result=result)


@pytest.mark.parametrize(
    "logging_level_str", ["jibberish", "", "hello?", "ERRORR", "C-C-CRITICAL"]
)
def test_fractopo_callback_error(logging_level_str: str):
    """
    Test fractopo_callback with invalid cli options.
    """
    with pytest.raises(SystemExit):
        result = typer_cli_runner.invoke(
            cli.APP,
            [
                "--log-level",
                logging_level_str,
                "info",
            ],
        )
        tests.click_error_print(result=result)

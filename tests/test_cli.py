from pathlib import Path
import geopandas as gpd
from click.testing import CliRunner
from fractopo import cli
import pytest
from fractopo.tval.trace_validator import BaseValidator
from fractopo.tval.executor_v2 import Validation
from tests import Helpers


@pytest.mark.parametrize(
    "trace_path, area_path, auto_fix", Helpers.test_tracevalidate_params
)
def test_tracevalidate(
    trace_path: Path, area_path: Path, auto_fix: str, tmp_path: Path
):
    """
    Tests tracevalidate click functionality.
    """
    clirunner = CliRunner()
    output_file = tmp_path / f"{trace_path.stem}.{trace_path.suffix}"
    cli_args = [str(trace_path), str(area_path), auto_fix, "--output", str(output_file)]
    result = clirunner.invoke(cli.tracevalidate, cli_args)
    # Check that exit code is 0 (i.e. ran succesfully.)
    assert result.exit_code == 0
    # Checks if output is saved
    assert output_file.exists()
    if not BaseValidator.ERROR_COLUMN in gpd.read_file(output_file).columns:
        assert "shp" in trace_path.suffix
        assert "VALID" in str(gpd.read_file(output_file).columns) or "valid" in str(
            gpd.read_file(output_file).columns
        )


@pytest.mark.parametrize(
    "trace_path, area_path, auto_fix", Helpers.test_tracevalidatev2_params
)
def test_tracevalidatev2(
    trace_path: Path, area_path: Path, auto_fix: str, tmp_path: Path
):
    """
    Tests tracevalidate click functionality.
    """
    clirunner = CliRunner()
    output_file = tmp_path / f"{trace_path.stem}.{trace_path.suffix}"
    cli_args = [
        str(trace_path),
        str(area_path),
        auto_fix,
        "--output",
        str(output_file),
        "--summary",
    ]
    result = clirunner.invoke(cli.tracevalidatev2, cli_args)
    # Check that exit code is 0 (i.e. ran succesfully.)
    assert result.exit_code == 0
    # Checks if output is saved
    assert output_file.exists()
    output_gdf = gpd.read_file(output_file)
    if not Validation.ERROR_COLUMN in output_gdf.columns:
        assert "shp" in trace_path.suffix
        assert "VALID" in str(output_gdf.columns) or "valid" in str(output_gdf.columns)
    if "--summary" in cli_args:
        assert "Out of" in result.output
        assert "There were" in result.output

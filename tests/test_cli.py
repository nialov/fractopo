from pathlib import Path
import geopandas as gpd
from click.testing import CliRunner
from fractopo import cli
import pytest
from fractopo.tval.trace_validator import BaseValidator
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
    # Checks if output path is printed
    assert output_file.exists()
    if not BaseValidator.ERROR_COLUMN in gpd.read_file(output_file).columns:
        assert "shp" in trace_path.suffix
        assert "VALID" in str(gpd.read_file(output_file).columns) or "valid" in str(
            gpd.read_file(output_file).columns
        )

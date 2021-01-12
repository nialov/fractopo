from pathlib import Path

from click.testing import CliRunner
from tests import Helpers

import geopandas as gpd
import pytest
from fractopo import cli
from fractopo.tval.trace_validation import Validation


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
    cli_args = [
        str(trace_path),
        str(area_path),
        auto_fix,
        "--output",
        str(output_file),
        "--summary",
        "--snap-threshold",
        "0.01",
    ]
    result = clirunner.invoke(cli.tracevalidate, cli_args)
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


def test_make_output_dir(tmp_path):
    some_file = Path(tmp_path) / "some.file"
    output_dir = cli.make_output_dir(some_file)
    assert output_dir.exists()
    assert output_dir.is_dir()

import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest

import tests

SAMPLE_DATA_DIR = Path(__file__).parent.parent.joinpath("sample_data/")
VALIDATION_NOTEBOOK = Path(__file__).parent.parent.parent.joinpath(
    "marimos/validation.py"
)
PYTHON_INTERPRETER = sys.executable

check_python_call = partial(
    subprocess.check_call,
    env={
        **os.environ,
        "PYTHONPATH": str.join(
            ":",
            [
                str(Path(__file__).parent.parent.parent),
                *sys.path,
            ],
        ),
    },
)


@pytest.mark.xfail(
    sys.platform == "win32",
    reason="Subprocess call is flaky in Windows",
    raises=subprocess.CalledProcessError,
)
@pytest.mark.parametrize(
    "traces_path,area_path,name",
    [
        (
            SAMPLE_DATA_DIR.joinpath("hastholmen_traces.geojson").as_posix(),
            SAMPLE_DATA_DIR.joinpath("hastholmen_area.geojson").as_posix(),
            "hastholmen",
        ),
        (
            tests.kb7_trace_100_path.as_posix(),
            tests.kb7_area_path.as_posix(),
            "kb7",
        ),
    ],
)
def test_validation_cli(traces_path: str, area_path: str, name: str):
    args = [
        "--traces-path",
        traces_path,
        "--area-path",
        area_path,
        "--name",
        name,
    ]

    check_python_call(
        [PYTHON_INTERPRETER, VALIDATION_NOTEBOOK.as_posix(), *args],
    )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Subprocess call is flaky in Windows",
)
@pytest.mark.parametrize(
    "args,raises",
    [
        (["--wrong-arg"], pytest.raises(subprocess.CalledProcessError)),
    ],
)
def test_validation_cli_args(args, raises):
    with raises:
        check_python_call([PYTHON_INTERPRETER, VALIDATION_NOTEBOOK.as_posix(), *args])

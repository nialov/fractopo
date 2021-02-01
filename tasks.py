"""
Invoke tasks.
"""
from pathlib import Path

from invoke import task


nox_tests = (
    "tests_strict",
    "tests_lazy",
    "test_tracevalidate",
    "rstcheck_docs",
    "docs",
)

conda_requirements_txt = Path("requirements-conda.txt")
requirements_txt = Path("docs_src/requirements.txt")


@task
def requirements(c):
    """
    Sync requirements from Pipfile to requirements*.txt and setup.py.
    """
    c.run(
        "pipenv run pipenv_to_requirements "
        "-o docs_src/requirements.txt -d docs_src/requirements-dev.txt"
    )
    c.run("nox --session pipenv_setup_sync")

    # Make custom conda requirements
    req_contents: str = requirements_txt.read_text()
    if not isinstance(req_contents, str):
        raise TypeError("Expected requirements.txt to have text contents.")
    # sklearn is named scikit-learn in conda
    req_contents = req_contents.replace("sklearn", "scikit-learn")
    conda_requirements_txt.write_text(req_contents)
    print("requirements-conda.txt successfully updated.")


@task
def nox(c):
    """
    Run whole nox suite synchronously.
    """
    c.run("nox --no-color")


@task
def nox_parallel(c):
    """
    Run nox suite in parallel.
    """
    promises = [
        c.run(
            f"nox --session {nox_test} --no-color",
            asynchronous=True,
            timeout=240,
        )
        for nox_test in nox_tests
    ]
    results = [promise.join() for promise in promises]
    for result in results:
        print(result)


@task
def format_notebooks(c):
    """
    Format complementary notebooks.
    """
    c.run("black-nb notebooks docs_src/notebooks")


@task(pre=[requirements, nox, format_notebooks])
def make(_):
    """
    Test and make everything.
    """


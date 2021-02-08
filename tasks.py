"""
Invoke tasks.
"""
from pathlib import Path

from invoke import task

nox_parallel_sessions = (
    "tests_strict",
    "tests_lazy",
    "test_tracevalidate",
    "docs",
)


conda_requirements_txt = Path("requirements-conda.txt")
requirements_txt = Path("docs_src/requirements.txt")


@task
def requirements(c):
    """
    Sync requirements from Pipfile to setup.py.
    """
    # Uses nox
    c.run("nox --session requirements")
    # # Make custom conda requirements
    # req_contents: str = requirements_txt.read_text()
    # if not isinstance(req_contents, str):
    #     raise TypeError("Expected requirements.txt to have text contents.")
    # # sklearn is named scikit-learn in conda
    # req_contents = req_contents.replace("sklearn", "scikit-learn")
    # conda_requirements_txt.write_text(req_contents)
    # print("requirements-conda.txt successfully updated.")


@task
def format(c):
    """
    Format everything.
    """
    c.run("nox --session format")


@task(pre=[format])
def lint(c):
    """
    Lint everything.
    """
    c.run("nox --session lint")


@task
def nox_parallel(c):
    """
    Run selected nox test suite sessions in parallel.
    """
    promises = [
        c.run(
            f"nox --session {nox_test} --no-color",
            asynchronous=True,
            timeout=360,
        )
        for nox_test in nox_parallel_sessions
    ]
    results = [promise.join() for promise in promises]
    for result in results:
        print(result)


@task(pre=[nox_parallel])
def test(_):
    """
    Run tests.
    """


@task
def docs(c):
    """
    Make documentation to docs using nox.
    """
    c.run("nox --session docs")


@task(pre=[test, lint, docs])
def make(_):
    """
    Make all.
    """

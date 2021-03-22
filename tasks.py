"""
Invoke tasks.

Most tasks employ nox to create a virtual session for testing.
"""
from pathlib import Path

from invoke import UnexpectedExit, task

nox_parallel_sessions = (
    "tests_pipenv",
    "tests_pip",
)

package_name = "fractopo"
coverage_badge_svg_path = Path("docs_src/imgs/coverage.svg")


@task
def requirements(c):
    """
    Sync requirements from Pipfile to setup.py.
    """
    c.run("nox --session requirements")


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


@task(pre=[requirements])
def nox_parallel(c):
    """
    Run selected nox test suite sessions in parallel.
    """
    # Run asynchronously and collect promises
    print(f"Running {len(nox_parallel_sessions)} nox test sessions.")
    promises = [
        c.run(
            f"nox --session {nox_test} --no-color",
            asynchronous=True,
            timeout=360,
        )
        for nox_test in nox_parallel_sessions
    ]

    # Join all promises
    results = [promise.join() for promise in promises]

    # Check if Result has non-zero exit code (should've already thrown error.)
    for result in results:
        if result.exited != 0:
            raise UnexpectedExit(result)

    # Report to user of success.
    print(f"{len(results)} nox sessions ran succesfully.")


@task(pre=[requirements])
def ci_test(c):
    """
    Test suite for continous integration testing.

    Installs with pip, tests with pytest and checks coverage with coverage.
    """
    c.run("nox --session tests_pip")


@task(pre=[nox_parallel])
def test(_):
    """
    Run tests.

    This is an extensive suite. It first tests in current environment and then
    creates virtual sessions with nox to test installation -> tests.
    """


@task(pre=[requirements])
def docs(c):
    """
    Make documentation to docs using nox.
    """
    c.run("nox --session docs")


@task(pre=[requirements, test, lint, docs])
def make(_):
    """
    Make all.
    """
    print("---------------")
    print("make successful.")


@task
def profile_network_runtime(c):
    """
    Profile Network analysis with pyinstrument.
    """
    c.run("nox --session profile_network_analysis")

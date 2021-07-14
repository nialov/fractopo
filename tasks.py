"""
Invoke tasks.

Most tasks employ nox to create a virtual session for testing.
"""

from invoke import UnexpectedExit, task

NOX_PARALLEL_SESSIONS = ("tests_pip",)

PACKAGE_NAME = "fractopo"


@task
def requirements(c):
    """
    Sync requirements.
    """
    c.run("nox --session requirements")


@task(pre=[requirements])
def format_and_lint(c):
    """
    Format and lint everything.
    """
    c.run("nox --session format_and_lint")


@task(pre=[requirements])
def nox_parallel(c):
    """
    Run selected nox test suite sessions in parallel.
    """
    # Run asynchronously and collect promises
    print(f"Running {len(NOX_PARALLEL_SESSIONS)} nox test sessions.")
    promises = [
        c.run(
            f"nox --session {nox_test} --no-color",
            asynchronous=True,
            timeout=360,
        )
        for nox_test in NOX_PARALLEL_SESSIONS
    ]

    # Join all promises
    results = [promise.join() for promise in promises]

    # Check if Result has non-zero exit code (should've already thrown error.)
    for result in results:
        if result.exited != 0:
            raise UnexpectedExit(result)

    # Report to user of success.
    print(f"{len(results)} nox sessions ran succesfully.")


@task
def update_version(c):
    """
    Update pyproject.toml version string.
    """
    c.run("nox --session update_version")


@task(pre=[requirements, update_version])
def ci_test(c):
    """
    Test suite for continous integration testing.

    Installs with pip, tests with pytest and checks coverage with coverage.
    """
    c.run("nox --session tests_pip")


@task(pre=[requirements, nox_parallel])
def test(_):
    """
    Run tests.

    This is an extensive suite. It first tests in current environment and then
    creates virtual sessions with nox to test installation -> tests.
    """


@task(pre=[requirements, update_version])
def docs(c):
    """
    Make documentation to docs using nox.
    """
    c.run("nox --session docs")


@task(pre=[requirements])
def notebooks(c):
    """
    Execute and fill notebooks.
    """
    c.run("nox --session notebooks")


@task(pre=[update_version, test, format_and_lint, docs, notebooks])
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

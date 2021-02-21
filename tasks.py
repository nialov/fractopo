"""
Invoke tasks.

Most tasks employ nox to create a virtual session for testing.
"""
from invoke import UnexpectedExit, task

nox_parallel_sessions = (
    "tests_strict",
    "tests_lazy",
)

package_name = "fractopo"


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


@task
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


@task
def pytest(c):
    """
    Run tests with pytest in currently installed environment.

    Much faster than full `test` suite.
    """
    c.run(f"coverage run --include '{package_name}/**.py' -m pytest")
    c.run("coverage report --fail-under 70")


@task(pre=[pytest, nox_parallel])
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


@task(pre=[test, lint, docs])
def make(_):
    """
    Make all.
    """
    print("---------------")
    print("make succesful.")

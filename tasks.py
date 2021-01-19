"""
Invoke tasks.
"""
from invoke import task


nox_tests = (
    "tests_strict",
    "tests_lazy",
    "test_tracevalidate",
    "rstcheck_docs",
    "docs",
)


@task
def requirements(c):
    """
    Sync requirements from Pipfile to requirements*.txt and setup.py.
    """
    c.run(
        "pipenv run pipenv_to_requirements "
        "-o docs_src/requirements.txt -d docs_src/requirements-dev.txt"
    )
    c.run("pipenv run pipenv-setup sync --pipfile --dev")


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
        c.run(f"nox --session {nox_test} --no-color", asynchronous=True, timeout=240,)
        for nox_test in nox_tests
    ]
    results = [promise.join() for promise in promises]
    for result in results:
        print(result)


@task(pre=[requirements, nox])
def make(_):
    """
    Test and make everything.
    """


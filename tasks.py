"""
Invoke tasks.

Most tasks employ nox to create a virtual session for testing.
"""
from invoke import task

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


@task
def update_version(c):
    """
    Update pyproject.toml and package/__init__.py version strings.
    """
    c.run("nox --session update_version")


@task(pre=[requirements, update_version])
def ci_test(c, python=""):
    """
    Test suite for continous integration testing.

    Installs with pip, tests with pytest and checks coverage with coverage.
    """
    python_version = "" if len(python) == 0 else f"-p {python}"
    c.run(f"nox --session tests_pip {python_version}")


@task(pre=[requirements, update_version])
def docs(c):
    """
    Make documentation to docs using nox.
    """
    print("Making documentation.")
    c.run("nox --session docs")


@task(pre=[requirements])
def notebooks(c):
    """
    Execute and fill notebooks.
    """
    print("Executing and filling notebooks.")
    c.run("nox --session notebooks")


@task(pre=[requirements, update_version])
def build(c):
    """
    Build package with poetry.
    """
    print("Building package with poetry.")
    c.run("nox --session build")


@task(pre=[requirements])
def typecheck(c):
    """
    Typecheck ``fractopo`` with ``mypy``.
    """
    print("Typechecking Python code with mypy.")
    c.run("nox --session typecheck")


@task(pre=[requirements])
def performance_profile(c):
    """
    Profile fractopo performance with ``pyinstrument``.
    """
    print("Profiling fractopo performance with pyinstrument.")
    c.run("nox --session profile_performance")


@task(pre=[format_and_lint, ci_test, build, docs])
def prepush(_):
    """
    Test suite for locally verifying continous integration results upstream.
    """


@task(
    pre=[
        update_version,
        format_and_lint,
        docs,
        notebooks,
        build,
        typecheck,
        performance_profile,
    ]
)
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

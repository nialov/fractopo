"""
Nox test suite.
"""

from pathlib import Path
from shutil import rmtree

import nox


docs_apidoc_dir_path = Path("docs_src/apidoc")
docs_dir_path = Path("docs")


@nox.session(python="3.8")
def tests_strict(session):
    """
    Run strict test suite.
    """
    session.install("pipenv")
    session.run(*"pipenv sync --dev --bare".split(" "))
    session.run(
        "pipenv",
        "run",
        "coverage",
        "run",
        "--include",
        "fractopo/**.py",
        "-m",
        "pytest",
    )
    session.run("pipenv", "run", "coverage", "report", "--fail-under", "70")


@nox.session(python="3.8")
def tests_lazy(session):
    """
    Run lazy test suite.
    """
    session.install(".[dev]")
    session.run("pytest")


@nox.session(python="3.8")
def test_tracevalidate(session):
    """
    Run test on tracevalidate script in new virtualenv.
    """
    session.install(".")
    # session.chdir(session.create_tmp())
    session.run("tracevalidate", "--help")


@nox.session(python="3.8")
def rstcheck_docs(session):
    """
    Check docs_src with rstcheck.
    """
    session.install("rstcheck", "sphinx")
    session.run(
        "rstcheck",
        "-r",
        "docs_src",
        "--ignore-directives",
        "automodule",
    )


@nox.session
def docs(session):
    """
    Make documentation.

    Installation mimics readthedocs install.
    """
    session.chdir(Path(__file__).parent)
    session.install(".[dev]")
    if docs_apidoc_dir_path.exists():
        rmtree(docs_apidoc_dir_path)
    if docs_dir_path.exists():
        rmtree(docs_dir_path)
    session.run("sphinx-apidoc", "-o", "./docs_src/apidoc", "./fractopo", "-e", "-f")
    session.run(
        "sphinx-build",
        "./docs_src",
        "./docs",
        "-b",
        "html",
    )


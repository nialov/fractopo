"""
Nox test suite.
"""

from pathlib import Path
from shutil import copy2, copytree, rmtree

import nox


docs_apidoc_dir_path = Path("docs_src/apidoc")
docs_dir_path = Path("docs")
package_name = "fractopo"
tests_name = "tests"
pipfile_lock = "Pipfile.lock"
notebooks_dir = "notebooks"


@nox.session(python="3.8")
def tests_strict(session: nox.Session):
    """
    Run strict test suite.
    """
    tmp_dir = session.create_tmp()
    for to_copy in (package_name, tests_name, pipfile_lock, notebooks_dir):
        if Path(to_copy).is_dir():
            copytree(to_copy, Path(tmp_dir) / to_copy)
        elif Path(to_copy).is_file():
            copy2(to_copy, tmp_dir)
        elif Path(to_copy).exists():
            ValueError("File not dir or file.")
        else:
            FileNotFoundError("Expected file to be found.")
    session.chdir(tmp_dir)
    session.install("pipenv")
    session.run("pipenv", "sync", "--python", f"{session.python}", "--dev", "--bare")
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
    # Test notebook(s)
    session.run("ipython", "notebooks/fractopo_network.ipynb")


@nox.session(python="3.8")
def tests_lazy(session):
    """
    Run lazy test suite.
    """
    session.install(".[dev]")
    # Test with pytest
    session.run("pytest")
    # Test notebook(s)
    session.run("ipython", "notebooks/fractopo_network.ipynb")


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
def pipenv_setup_sync(session):
    """
    Sync Pipfile to setup.py with pipenv-setup.
    """
    session.install("pipenv-setup")
    session.run("pipenv-setup", "sync", "--pipfile", "--dev")


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


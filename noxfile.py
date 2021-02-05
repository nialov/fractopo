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
notebooks_name = "notebooks"
importanize_config = "importanize.ini"
tasks_name = "tasks.py"
noxfile_name = "noxfile.py"
pylama_config = "pylama.ini"


@nox.session(python="3.8")
def tests_strict(session: nox.Session):
    """
    Run strict test suite.
    """
    tmp_dir = session.create_tmp()
    for to_copy in (package_name, tests_name, pipfile_lock, notebooks_name):
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
def format(session):
    """
    Format python files, notebooks and docs_src.
    """
    session.install("black", "black-nb", "importanize")
    # Format python files
    session.run("black", package_name, tests_name, tasks_name, noxfile_name)
    # Format python file imports
    session.run(
        "importanize",
        package_name,
        tests_name,
        tasks_name,
        noxfile_name,
    )
    # Format notebooks
    session.run("black-nb", notebooks_name)


@nox.session(python="3.8")
def lint(session):
    """
    Lint python files, notebooks and docs_src.
    """
    session.install("rstcheck", "sphinx", "black", "black-nb", "importanize", "pylama")
    # Lint docs
    session.run(
        "rstcheck",
        "-r",
        "docs_src",
        "--ignore-directives",
        "automodule",
    )
    # Lint python files with black (all should be formatted.)
    session.run("black", "--check", package_name, tests_name, tasks_name, noxfile_name)
    session.run(
        "importanize",
        "--ci",
        package_name,
        tests_name,
        tasks_name,
        noxfile_name,
    )
    # Lint with pylama
    session.run(
        "pylama",
        "-o",
        pylama_config,
        package_name,
        tests_name,
        tasks_name,
        noxfile_name,
    )

    # Lint notebooks with black-nb (all should be formatted.)
    session.run("black-nb", "--check", notebooks_name)


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

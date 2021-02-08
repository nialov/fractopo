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
tasks_name = "tasks.py"
noxfile_name = "noxfile.py"
pylama_config = "pylama.ini"


docs_notebooks = Path("docs_src/notebooks").glob("*.ipynb")
regular_notebooks = Path(notebooks_name).glob("*.ipynb")


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
    session.run("pipenv", "--rm")
    session.run(
        "pipenv",
        "sync",
        "--python",
        f"{session.python}",
        "--dev",
        "--bare",
    )
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
    # Test with pytest
    session.run("pytest")
    # Test notebook(s)
    for notebook in regular_notebooks:
        session.run("ipython", str(notebook))


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
    session.install("black", "black-nb", "isort")
    # Format python files
    session.run("black", package_name, tests_name, tasks_name, noxfile_name)
    # Format python file imports
    session.run(
        "isort",
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
    session.install("rstcheck", "sphinx", "black", "black-nb", "isort", "pylama")
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
        "isort",
        "--check-only",
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
def requirements(session):
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

    # Install with setup.py[dev] installation
    session.install(".[dev]")

    # Remove old apidocs
    if docs_apidoc_dir_path.exists():
        rmtree(docs_apidoc_dir_path)

    # Remove all old docs
    if docs_dir_path.exists():
        rmtree(docs_dir_path)

    # Execute and fill cells in docs notebooks
    for notebook in docs_notebooks:
        session.run(
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--inplace",
            "--execute",
            str(notebook),
        )
    # Create apidocs
    session.run("sphinx-apidoc", "-o", "./docs_src/apidoc", "./fractopo", "-e", "-f")

    # Create docs in ./docs folder
    session.run(
        "sphinx-build",
        "./docs_src",
        "./docs",
        "-b",
        "html",
    )

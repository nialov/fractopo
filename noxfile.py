"""
Nox test suite.
"""
import logging
from pathlib import Path
from shutil import copy2, copytree, rmtree
from typing import List

import nox

# Variables
package_name = "fractopo"

# Paths
docs_apidoc_dir_path = Path("docs_src/apidoc")
docs_dir_path = Path("docs")
coverage_svg_path = Path("docs_src/imgs/coverage.svg")

# Path strings
tests_name = "tests"
pipfile_lock = "Pipfile.lock"
notebooks_name = "notebooks"
docs_notebooks_name = "docs_src/notebooks"
tasks_name = "tasks.py"
noxfile_name = "noxfile.py"
pylama_config = "pylama.ini"

# Globs
docs_notebooks = Path("docs_src/notebooks").glob("*.ipynb")
regular_notebooks = Path(notebooks_name).glob("*.ipynb")
all_notebooks = list(docs_notebooks) + list(regular_notebooks)


def filter_paths_to_existing(*iterables) -> List[str]:
    """
    Filter paths to only existing.
    """
    return [str(path) for path in iterables if Path(path).exists()]


def fill_notebook(session, notebook: Path):
    """
    Execute and fill notebook outputs.
    """
    session.run(
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--inplace",
        "--execute",
        str(notebook),
    )


@nox.session(python="3.8")
def tests_pipenv(session: nox.Session):
    """
    Run test suite with pipenv sync.
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
            logging.error(f"Expected {to_copy} to exist.")
    session.chdir(tmp_dir)
    session.install("pipenv")
    session.run("pipenv", "--rm", success_codes=[0, 1])
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
        "pytest",
    )


@nox.session(python="3.8")
def tests_pip(session):
    """
    Run test suite with pip install.
    """
    # Install dependencies dev + coverage
    session.install(".[dev,coverage]")

    # Test with pytest and determine coverage
    session.run("coverage", "run", "--source", package_name, "-m", "pytest")

    # Fails with test coverage under 70
    session.run("coverage", "report", "--fail-under", "70")

    # Make coverage-badge image
    if coverage_svg_path.exists():
        coverage_svg_path.unlink()
    elif not coverage_svg_path.parent.exists():
        coverage_svg_path.parent.mkdir(parents=True)
    session.run("coverage-badge", "-o", str(coverage_svg_path))


@nox.session(python="3.8")
def notebooks(session):
    """
    Run notebooks.

    Notebooks are usually run in remote so use pip install.
    Note that notebooks shouldn't have side effects i.e. file disk file writing.
    """
    session.install(".[dev]")
    # Test notebook(s)
    for notebook in all_notebooks:
        fill_notebook(session=session, notebook=notebook)


@nox.session(python="3.8")
def format(session):
    """
    Format Python files, notebooks and docs_src.
    """
    # Install only format dependencies
    session.install("black", "black-nb", "isort")
    existing_paths = filter_paths_to_existing(
        package_name, tests_name, tasks_name, noxfile_name
    )

    # Format python files
    session.run("black", *existing_paths)

    # Format python file imports
    session.run(
        "isort",
        *existing_paths,
    )

    # Format notebooks
    for notebook in all_notebooks:
        session.run("black-nb", str(notebook))


@nox.session(python="3.8")
def lint(session):
    """
    Lint python files, notebooks and docs_src.
    """
    # Install only lint dependencies
    session.install("rstcheck", "sphinx", "black", "black-nb", "isort", "pylama")
    existing_paths = filter_paths_to_existing(
        package_name, tests_name, tasks_name, noxfile_name
    )

    # Lint docs
    session.run(
        "rstcheck",
        "-r",
        "docs_src",
        "--ignore-directives",
        "automodule",
    )

    # Lint Python files with black (all should be formatted.)
    session.run("black", "--check", *existing_paths)
    session.run(
        "isort",
        "--check-only",
        *existing_paths,
    )

    # Lint with pylama
    session.run(
        "pylama",
        "-o",
        pylama_config,
        *existing_paths,
    )

    for notebook in all_notebooks:
        # Lint notebooks with black-nb (all should be formatted.)
        session.run("black-nb", "--check", str(notebook))


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
    session.install(".[dev,docs]")

    # Remove old apidocs
    if docs_apidoc_dir_path.exists():
        rmtree(docs_apidoc_dir_path)

    # Remove all old docs
    if docs_dir_path.exists():
        rmtree(docs_dir_path)

    # Execute and fill cells in docs notebooks
    for notebook in docs_notebooks:
        fill_notebook(session=session, notebook=notebook)

    # Create apidocs
    session.run(
        "sphinx-apidoc", "-o", "./docs_src/apidoc", f"./{package_name}", "-e", "-f"
    )

    # Create docs in ./docs folder
    session.run(
        "sphinx-build",
        "./docs_src",
        "./docs",
        "-b",
        "html",
    )


@nox.session(reuse_venv=True)
def profile_network_analysis(session):
    """
    Profile Network analysis with pyinstrument.
    """
    # Install with setup.py[dev] installation
    session.install(".[dev]", "pyinstrument")

    # Run pyprofiler
    session.run(
        "pyinstrument",
        "--renderer",
        "html",
        "--outfile",
        "tests/profile_runtime.html",
        "tests/profile_runtime.py",
    )

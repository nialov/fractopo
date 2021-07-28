"""
Nox test suite.
"""
from pathlib import Path
from shutil import rmtree
from typing import List

import nox

# Variables
PACKAGE_NAME = "fractopo"

# Paths
DOCS_APIDOC_DIR_PATH = Path("docs_src/apidoc")
DOCS_DIR_PATH = Path("docs")
COVERAGE_SVG_PATH = Path("docs_src/imgs/coverage.svg")
PROFILE_SCRIPT_PATH = Path("tests/_profile.py")

# Path strings
TESTS_NAME = "tests"
NOTEBOOKS_NAME = "notebooks"
TASKS_NAME = "tasks.py"
NOXFILE_NAME = "noxfile.py"
DEV_REQUIREMENTS = "requirements.txt"
DOCS_REQUIREMENTS = "docs_src/requirements.txt"
DOCS_EXAMPLES = "examples"
DOCS_AUTO_EXAMPLES = "docs_src/auto_examples"

# Globs
DOCS_NOTEBOOKS = Path("docs_src/notebooks").glob("*.ipynb")
REGULAR_NOTEBOOKS = Path(NOTEBOOKS_NAME).glob("*.ipynb")
ALL_NOTEBOOKS = list(DOCS_NOTEBOOKS) + list(REGULAR_NOTEBOOKS)

PYTHON_VERSIONS = ["3.7", "3.8", "3.9"]


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


def install_dev(session, extras: str = ""):
    """
    Install all package and dev dependencies.
    """
    session.install(f".{extras}")
    session.install("-r", DEV_REQUIREMENTS)


@nox.session(python=PYTHON_VERSIONS)
def tests_pip(session):
    """
    Run test suite with pip install.
    """
    # Check if any tests exist
    tests_path = Path(TESTS_NAME)
    if (
        (not tests_path.exists())
        or (tests_path.is_file())
        or (len(list(tests_path.iterdir())) == 0)
    ):
        print("No tests in {TESTS_NAME} directory.")
        return

    # Install dependencies dev + coverage
    install_dev(session=session, extras="[coverage]")

    # Test with pytest and determine coverage
    session.run("coverage", "run", "--source", PACKAGE_NAME, "-m", "pytest")

    # Fails with test coverage under 70
    session.run("coverage", "report", "--fail-under", "70")

    # Make coverage-badge image
    if COVERAGE_SVG_PATH.exists():
        COVERAGE_SVG_PATH.unlink()
    elif not COVERAGE_SVG_PATH.parent.exists():
        COVERAGE_SVG_PATH.parent.mkdir(parents=True)
    session.run("coverage-badge", "-o", str(COVERAGE_SVG_PATH))


@nox.session(python=PYTHON_VERSIONS)
def notebooks(session):
    """
    Run notebooks.

    Notebooks are usually run in remote so use pip install.
    Note that notebooks shouldn't have side effects i.e. disk file writing.
    """
    # Check if any notebooks exist.
    if len(ALL_NOTEBOOKS) == 0:
        print("No notebooks found.")
        return

    # Install dev dependencies
    install_dev(session=session)

    # Test notebook(s)
    for notebook in ALL_NOTEBOOKS:
        fill_notebook(session=session, notebook=notebook)


@nox.session
def format_and_lint(session):
    """
    Format and lint python files, notebooks and docs_src.
    """
    existing_paths = filter_paths_to_existing(
        PACKAGE_NAME, TESTS_NAME, TASKS_NAME, NOXFILE_NAME, DOCS_EXAMPLES
    )

    if len(existing_paths) == 0:
        print("Nothing to format or lint.")
        return

    # Install formatting and lint dependencies
    install_dev(session=session, extras="[format-lint]")

    # Format python files
    session.run("black", *existing_paths)

    # Format python file imports
    session.run(
        "isort",
        *existing_paths,
    )

    # Format notebooks
    for notebook in ALL_NOTEBOOKS:
        session.run("black-nb", str(notebook))

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

    # Lint with pylint
    session.run(
        "pylint",
        *existing_paths,
    )

    for notebook in ALL_NOTEBOOKS:
        # Lint notebooks with black-nb (all should be formatted.)
        session.run("black-nb", "--check", str(notebook))


@nox.session
def requirements(session):
    """
    Sync poetry requirements from pyproject.toml to requirements.txt.
    """
    # Install poetry
    session.install("poetry")

    # Sync dev requirements
    session.run("poetry", "export", "--without-hashes", "--dev", "-o", DEV_REQUIREMENTS)

    # Sync docs requirements
    session.run(
        "poetry",
        "export",
        "--without-hashes",
        "--dev",
        "-E",
        "docs",
        "-o",
        DOCS_REQUIREMENTS,
    )


@nox.session(reuse_venv=True)
def docs(session):
    """
    Make documentation.

    Installation mimics readthedocs install.
    """
    # Install from docs_src/requirements.txt that has been synced with docs
    # requirements
    session.install(".")
    session.install("-r", DOCS_REQUIREMENTS)

    # Remove old apidocs
    if DOCS_APIDOC_DIR_PATH.exists():
        rmtree(DOCS_APIDOC_DIR_PATH)

    # Remove all old docs
    if DOCS_DIR_PATH.exists():
        rmtree(DOCS_DIR_PATH)

    # Execute and fill cells in docs notebooks
    for notebook in DOCS_NOTEBOOKS:
        fill_notebook(session=session, notebook=notebook)

    # Create apidocs
    session.run(
        "sphinx-apidoc", "-o", "./docs_src/apidoc", f"./{PACKAGE_NAME}", "-e", "-f"
    )

    try:
        # Create docs in ./docs folder
        session.run(
            "sphinx-build",
            "./docs_src",
            "./docs",
            "-b",
            "html",
        )

    finally:
        # Clean up sphinx-gallery folder in ./docs_src/auto_examples
        auto_examples_path = Path(DOCS_AUTO_EXAMPLES)
        if auto_examples_path.exists():
            rmtree(auto_examples_path)


@nox.session
def update_version(session):
    """
    Update package version from git vcs.
    """
    # Install poetry-dynamic-versioning
    session.install("poetry-dynamic-versioning")

    # Run poetry-dynamic-versioning to update version tag in pyproject.toml
    # and fractopo/__init__.py
    session.run("poetry-dynamic-versioning")


@nox.session(python=PYTHON_VERSIONS)
def build(session):
    """
    Build package with poetry.
    """
    # Install poetry
    session.install("poetry")

    # Install dependencies to poetry
    session.run("poetry", "install")

    # Build
    session.run("poetry", "build")


@nox.session(reuse_venv=True)
def profile_performance(session):
    """
    Profile fractopo runtime performance.

    User must implement the actual performance utility.
    """
    # Install dev and pyinstrument
    install_dev(session)
    session.install("pyinstrument")

    # Create temporary path
    save_file = f"{session.create_tmp()}/profile_runtime.html"

    if not PROFILE_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            f"Expected {PROFILE_SCRIPT_PATH} to exist for performance profiling."
        )

    # Run pyprofiler
    session.run(
        "pyinstrument",
        "--renderer",
        "html",
        "--outfile",
        save_file,
        str(PROFILE_SCRIPT_PATH),
    )

    resolved_path = Path(save_file).resolve()
    print(f"\nPerformance profile saved at {resolved_path}.")


@nox.session
def typecheck(session):
    """
    Typecheck Python code.
    """
    existing_paths = filter_paths_to_existing(PACKAGE_NAME)

    if len(existing_paths) == 0:
        print("Nothing to typecheck.")
        return

    # Install package and typecheck dependencies
    install_dev(session=session, extras="[typecheck]")

    # Format python files
    session.run("mypy", *existing_paths)

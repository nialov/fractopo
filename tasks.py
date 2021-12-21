"""
Invoke tasks.

Most tasks employ nox to create a virtual session for testing.
"""
import re
from itertools import chain
from pathlib import Path
from time import strftime

from invoke import task

PACKAGE_NAME = "fractopo"
CITATION_CFF_PATH = Path("CITATION.cff")
DATE_RELEASED_STR = "date-released"
UTF8 = "utf-8"

VERSION_GLOBS = [
    "*/__init__.py",
    "CITATION.cff",
    "pyproject.toml",
]

VERSION_PATTERN = r"(^_*version_*\s*[:=]\s\").*\""


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
def docs(c, auto_build=False):
    """
    Make documentation to docs using nox.
    """
    print("Making documentation.")
    docs_session = "docs" if not auto_build else "auto_docs"
    if auto_build:
        print("Starting sphinx-autobuild service to watch for src changes.")
    c.run(f"nox --session {docs_session}")


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


@task
def citation(c):
    """
    Sync and validate CITATION.cff.
    """
    print("Updating CITATION.cff date")
    citation_text = CITATION_CFF_PATH.read_text(encoding=UTF8)
    citation_lines = citation_text.splitlines()
    if DATE_RELEASED_STR not in citation_text:
        raise ValueError(
            f"Expected to find {DATE_RELEASED_STR} str in {CITATION_CFF_PATH}."
            f"\nCheck & validate {CITATION_CFF_PATH}."
        )
    date = strftime("%Y-%m-%d")
    new_lines = [
        line if "date-released" not in line else f'date-released: "{date}"'
        for line in citation_lines
    ]
    CITATION_CFF_PATH.write_text("\n".join(new_lines), encoding=UTF8)

    print("Validating CITATION.cff")
    c.run("nox --session validate_citation_cff")


@task
def changelog(c, latest_version=""):
    """
    Generate changelog.
    """
    c.run(f"nox --session changelog -- {latest_version}")


@task
def codespell(c):
    """
    Check code spelling.
    """
    c.run("nox --session codespell")


@task(
    pre=[
        requirements,
        update_version,
        format_and_lint,
        ci_test,
        build,
        docs,
        citation,
        changelog,
        codespell,
    ]
)
def prepush(_):
    """
    Test suite for locally verifying continous integration results upstream.
    """


@task
def pre_commit(c, only_run=False, only_install=False):
    """
    Verify that pre-commit is installed, install its hooks and run them.
    """
    cmd = "pre-commit --help"
    try:
        c.run(cmd, hide=True)
    except Exception:
        print(f"Could not run '{cmd}'. Make sure pre-commit is installed.")
        raise

    if not only_run:
        c.run("pre-commit install")
        c.run("pre-commit install --hook-type commit-msg")
        print("Hooks installed!")

    if not only_install:
        print("Running on all files.")
        try:
            c.run("pre-commit run --all-files")
        except Exception:
            print("pre-commit run formatted files!")


@task(pre=[prepush], post=[pre_commit])
def tag(c, tag="", annotation=""):
    """
    Make new tag and update version strings accordingly
    """
    if len(tag) == 0:
        raise ValueError("Tag string must be specified with '--tag=*'.")
    if len(annotation) == 0:
        raise ValueError("Annotation string must be specified with '--annotation=*'.")

    # Create changelog with 'tag' as latest version
    c.run(f"nox --session changelog -- {tag}")

    # Remove v at the start of tag
    tag = tag if "v" not in tag else tag[1:]

    # Iterate over all files determined from VERSION_GLOBS
    for path in chain(*[Path(".").glob(glob) for glob in (VERSION_GLOBS)]):

        # Collect new lines
        new_lines = []
        for line in path.read_text(UTF8).splitlines():

            # Substitute lines with new tag if they match pattern
            substituted = re.sub(VERSION_PATTERN, r"\g<1>" + tag + r'"', line)

            # Report to user
            if line != substituted:
                print(
                    f"Replacing version string:\n{line}\nin"
                    f" {path} with:\n{substituted}\n"
                )
                new_lines.append(substituted)
            else:
                # No match, append line anyway
                new_lines.append(line)

        # Write results to files
        path.write_text("\n".join(new_lines), encoding=UTF8)

    cmds = (
        "# Run pre-commit to check files.",
        "pre-commit run --all-files",
        "git add .",
        "# Make sure only version updates are committed!",
        "git commit -m 'docs: update version'",
        "# Make sure tag is proper!",
        f"git tag -a v{tag} -m '{annotation}'",
    )
    print("Not running git cmds. See below for suggested commands:\n---\n")
    for cmd in cmds:
        print(cmd)


@task(
    pre=[
        prepush,
        notebooks,
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

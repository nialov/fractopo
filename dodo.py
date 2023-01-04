"""
doit tasks.

Most tasks employ nox to create a virtual session for testing.
"""
import re
from pathlib import Path
from time import strftime

from doit import task_params
from doit.tools import config_changed

# Strings
PACKAGE_NAME = "fractopo"
DATE_RELEASED_STR = "date-released"
UTF8 = "utf-8"
VERSION_PATTERN = r"(^_*version_*\s*[:=]\s\").*\""
ACTIONS = "actions"
FILE_DEP = "file_dep"
TASK_DEP = "task_dep"
TARGETS = "targets"
NAME = "name"
PARAMS = "params"
UP_TO_DATE = "uptodate"
PYTHON_VERSIONS = ["3.8", "3.9", "3.10"]
DEFAULT_PYTHON_VERSION = "3.8"

# Paths

## Docs
DOCS_SRC_PATH = Path("docs_src")
DOCS_PATH = Path("docs")
DOCS_EXAMPLES_PATH = Path("examples")
FRACTOPO_WORKFLOW_VISUALISATION_SCRIPT = (
    DOCS_EXAMPLES_PATH / "fractopo_workflow_visualisation.py"
)
FRACTOPO_WORKFLOW_VISUALISATION_PLOT = (
    DOCS_SRC_PATH / "imgs/fractopo_workflow_visualisation.jpg"
)
DOCS_REQUIREMENTS_PATH = Path("docs_src/requirements.txt")
NOTEBOOKS_PATH = DOCS_SRC_PATH / "notebooks"
COVERAGE_SVG_PATH = DOCS_SRC_PATH / Path("imgs/coverage.svg")
README_PATH = Path("README.rst")
# DOCS_FILES = [*list(DOCS_SRC_PATH.rglob("*.rst")), README_PATH]
DOCS_FILES = [
    *[
        path
        for path in DOCS_SRC_PATH.rglob("*.rst")
        # Filter out sphinx-gallery files
        if "auto_examples" not in str(path)
    ],
    README_PATH,
]
DOCS_APIDOC_PATH = DOCS_SRC_PATH / "apidoc"

## Build
DEV_REQUIREMENTS_PATH = Path("requirements.txt")
PYPROJECT_PATH = Path("pyproject.toml")
PRE_COMMIT_CONFIG_PATH = Path(".pre-commit-config.yaml")
POETRY_LOCK_PATH = Path("poetry.lock")
NOXFILE_PATH = Path("noxfile.py")
PACKAGE_INIT_PATH = Path(PACKAGE_NAME) / "__init__.py"
DODO_PATH = Path("dodo.py")

## Tests
TESTS_PATH = Path("tests")

## Misc
CITATION_CFF_PATH = Path("CITATION.cff")
VERSION_PATHS = [
    f"{PACKAGE_NAME}/__init__.py",
    CITATION_CFF_PATH,
    PYPROJECT_PATH,
]
CHANGELOG_PATH = Path("CHANGELOG.md")


PYTHON_SRC_FILES = [
    path for path in Path(PACKAGE_NAME).rglob("*.py") if "__init__" not in path.name
]
PYTHON_TEST_FILES = list(TESTS_PATH.rglob("*.py"))

PYTHON_UTIL_FILES = [
    *list(DOCS_EXAMPLES_PATH.rglob("*.py")),
    *list(Path(".").glob("*.py")),
]
PYTHON_ALL_FILES = [*PYTHON_SRC_FILES, *PYTHON_TEST_FILES, *PYTHON_UTIL_FILES]
NOTEBOOKS = [
    *list(NOTEBOOKS_PATH.rglob("*.ipynb")),
]
DIST_DIR_PATH = Path("dist/")


def resolve_task_name(func) -> str:
    """
    Resolve name of task without ``task_`` prefix.
    """
    return func.__name__.replace("task_", "")


def task_lock_check():
    """
    Check that poetry.lock is up to date with pyproject.toml.
    """
    # command = "nox --session requirements"
    cmd_list = [
        "poetry",
        "lock",
        "--check",
    ]
    cmd = " ".join(cmd_list)
    return {
        FILE_DEP: [POETRY_LOCK_PATH, PYPROJECT_PATH],
        ACTIONS: [cmd],
    }


def task_requirements():
    """
    Sync requirements from poetry.lock.
    """
    # command = "nox --session requirements"
    cmd_list = [
        "poetry",
        "export",
        "--without-hashes",
        "--dev",
        "{}",
        "-o",
        "{}",
    ]
    command_base = " ".join(cmd_list)
    for requirements_path, options in zip(
        (DEV_REQUIREMENTS_PATH, DOCS_REQUIREMENTS_PATH), ("", "-E docs")
    ):
        yield {
            NAME: str(requirements_path),
            FILE_DEP: [POETRY_LOCK_PATH],
            TASK_DEP: [resolve_task_name(task_lock_check)],
            ACTIONS: [command_base.format(options, requirements_path)],
            TARGETS: [requirements_path],
            UP_TO_DATE: [config_changed(dict(command_base=command_base))],
        }


def task_pre_commit():
    """
    Run pre-commit.
    """
    command = "nox --session pre_commit"
    return {
        ACTIONS: [command],
        TASK_DEP: [resolve_task_name(task_requirements)],
    }


def task_lint():
    """
    Lint everything.
    """
    command = "nox --session lint"
    return {
        FILE_DEP: [
            *PYTHON_ALL_FILES,
            *NOTEBOOKS,
            *DOCS_FILES,
            DEV_REQUIREMENTS_PATH,
            NOXFILE_PATH,
            # DODO_PATH,
        ],
        ACTIONS: [command],
        TASK_DEP: [resolve_task_name(task_pre_commit)],
        UP_TO_DATE: [config_changed(dict(command=command))],
    }


def task_ci_test():
    """
    Test suite for continuous integration testing.

    Installs with pip, tests with pytest and checks coverage with coverage.
    """
    # python_version = "" if len(python) == 0 else f"-p {python}"
    command_base = "nox --session tests_pip -p {}"
    for python_version in PYTHON_VERSIONS:
        command = command_base.format(python_version)
        yield {
            NAME: python_version,
            FILE_DEP: [
                *PYTHON_SRC_FILES,
                *PYTHON_TEST_FILES,
                DEV_REQUIREMENTS_PATH,
                # DODO_PATH,
            ],
            TASK_DEP: [resolve_task_name(task_pre_commit)],
            ACTIONS: [command],
            UP_TO_DATE: [config_changed(dict(command_base=command_base))],
            **(
                {TARGETS: [COVERAGE_SVG_PATH]}
                if python_version == DEFAULT_PYTHON_VERSION
                else dict()
            ),
        }


def task_apidocs():
    """
    Make apidoc documentation.
    """
    command = "nox --session apidocs"
    return {
        ACTIONS: [command],
        FILE_DEP: [
            *PYTHON_ALL_FILES,
            *DOCS_FILES,
            DOCS_REQUIREMENTS_PATH,
            NOXFILE_PATH,
            # DODO_PATH,
        ],
        TASK_DEP: [
            resolve_task_name(task_pre_commit),
            # resolve_task_name(task_update_version),
            resolve_task_name(task_lint),
        ],
        TARGETS: [DOCS_APIDOC_PATH],
        UP_TO_DATE: [config_changed(dict(command=command))],
    }


def task_apidocs():
    """
    Make apidoc documentation.
    """
    command = "nox --session apidocs"
    return {
        ACTIONS: [command],
        FILE_DEP: [
            *PYTHON_ALL_FILES,
            *DOCS_FILES,
            DOCS_REQUIREMENTS_PATH,
            NOXFILE_PATH,
            # DODO_PATH,
        ],
        TASK_DEP: [
            resolve_task_name(task_pre_commit),
            # resolve_task_name(task_update_version),
            resolve_task_name(task_lint),
        ],
        TARGETS: [DOCS_APIDOC_PATH],
        UP_TO_DATE: [config_changed(dict(command=command))],
    }


def task_docs():
    """
    Make documentation to docs using nox.
    """
    command = "nox --session docs"
    return {
        ACTIONS: [command],
        FILE_DEP: [
            *PYTHON_ALL_FILES,
            *DOCS_FILES,
            DOCS_REQUIREMENTS_PATH,
            NOXFILE_PATH,
            # DODO_PATH,
        ],
        TASK_DEP: [
            resolve_task_name(task_pre_commit),
            resolve_task_name(task_lint),
            # resolve_task_name(task_update_version),
            resolve_task_name(task_apidocs),
        ],
        TARGETS: [DOCS_PATH],
        UP_TO_DATE: [config_changed(dict(command=command))],
    }


def task_notebooks():
    """
    Execute and fill notebooks.
    """
    command = "nox --session notebooks"
    return {
        FILE_DEP: [
            *PYTHON_SRC_FILES,
            *NOTEBOOKS,
            DEV_REQUIREMENTS_PATH,
            NOXFILE_PATH,
            # DODO_PATH,
        ],
        TASK_DEP: [resolve_task_name(task_pre_commit)],
        ACTIONS: [command],
        UP_TO_DATE: [config_changed(dict(command=command))],
    }


def task_build():
    """
    Build package with poetry.

    Runs always without strict dependencies or targets.
    """
    return {
        ACTIONS: ["poetry build"],
    }


def task_typecheck():
    """
    Typecheck fractopo with ``mypy``.
    """
    command = "nox --session typecheck"
    # command = "nox --session build"
    return {
        ACTIONS: [command],
        FILE_DEP: [
            *PYTHON_SRC_FILES,
            DEV_REQUIREMENTS_PATH,
            NOXFILE_PATH,
            # DODO_PATH,
        ],
        TASK_DEP: [resolve_task_name(task_pre_commit)],
        UP_TO_DATE: [config_changed(dict(command=command))],
    }


def task_performance_profile():
    """
    Profile fractopo performance with ``pyinstrument``.
    """
    command = "nox --session profile_performance"
    # command = "nox --session build"
    return {
        ACTIONS: [command],
        FILE_DEP: [
            *PYTHON_SRC_FILES,
            *PYTHON_TEST_FILES,
            DEV_REQUIREMENTS_PATH,
            NOXFILE_PATH,
            # DODO_PATH,
        ],
        UP_TO_DATE: [config_changed(dict(command=command))],
    }


def update_citation():
    """
    Sync CITATION.cff.
    """
    citation_text = CITATION_CFF_PATH.read_text(UTF8)
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
    # new_lines.append("\n")

    # Write back to CITATION.cff including newline at end
    with CITATION_CFF_PATH.open("w", newline="\n", encoding=UTF8) as openfile:
        openfile.write("\n".join(new_lines) + "\n")


def task_codespell():
    """
    Check code spelling.
    """
    command = "nox --session codespell"
    return {
        ACTIONS: [command],
        FILE_DEP: [
            *PYTHON_ALL_FILES,
            POETRY_LOCK_PATH,
            NOXFILE_PATH,
            # DODO_PATH,
        ],
        TASK_DEP: [resolve_task_name(task_pre_commit)],
        UP_TO_DATE: [config_changed(dict(command=command))],
    }


def parse_tag(tag: str) -> str:
    """
    Parse numeric tag.

    E.g. v0.0.1 -> 0.0.1
    """
    return tag if "v" not in tag else tag[1:]


def use_tag(tag: str):
    """
    Use tag as version number in files.
    """
    assert len(tag) != 0
    tag = parse_tag(tag)
    # Remove v at the start of tag

    def replace_version_string(path: Path, tag: str):
        """
        Replace version string in file at path.
        """
        # Collect new lines
        new_lines = []
        for line in path.read_text(UTF8).splitlines():

            # Substitute lines with new tag if they match pattern
            substituted = re.sub(VERSION_PATTERN, r"\g<1>" + tag + r'"', line)

            # Report to user
            if line != substituted:
                print(
                    f"Replacing version string:\n{line}\n"
                    f"in {path} with:\n{substituted}\n"
                )
                new_lines.append(substituted)
            else:
                # No match, append line anyway
                new_lines.append(line)

        new_lines.append("\n")
        # Write results to file
        with path.open("w", newline="\n", encoding=UTF8) as openfile:
            openfile.write("\n".join(new_lines))

    # Iterate over all files determined from VERSION_PATHS
    for path_name in VERSION_PATHS:
        path = Path(path_name)
        replace_version_string(path=path, tag=tag)

    cmds = (
        "# Run pre-commit to check files.",
        "pre-commit run --all-files",
        "git add .",
        "# Make sure only version updates are committed!",
        "git commit -m 'docs: update version'",
        "# Make sure tag is proper and add annotation as wanted.",
        f"git tag -a v{tag} -m 'Release {tag}.'",
    )
    print("Not running git cmds. See below for suggested commands:\n---\n")
    for cmd in cmds:
        print(cmd)


@task_params([{NAME: "tag", "default": "", "type": str, "long": "tag"}])
def task_tag(tag: str):
    """
    Make new tag and update version strings accordingly
    """
    assert isinstance(tag, str)
    # Create changelog with 'tag' as latest version
    create_changelog = "nox --session changelog -- %(tag)s"
    return {
        ACTIONS: [create_changelog, update_citation, use_tag],
    }


def task_git_clean():
    """
    Clean all vcs untracked files with git.
    """
    cmd = "git clean -f -f -x -d"

    return {
        ACTIONS: [cmd],
    }


def task_create_workflow_visualisation():
    """
    Create ``fractopo`` workflow visualisation.
    """
    commands = [
        f"python {FRACTOPO_WORKFLOW_VISUALISATION_SCRIPT} {FRACTOPO_WORKFLOW_VISUALISATION_PLOT} ",
    ]
    return {
        ACTIONS: commands,
        FILE_DEP: [
            *PYTHON_ALL_FILES,
            POETRY_LOCK_PATH,
            FRACTOPO_WORKFLOW_VISUALISATION_SCRIPT,
        ],
        TASK_DEP: [resolve_task_name(task_requirements)],
        UP_TO_DATE: [config_changed(dict(commands=commands))],
        TARGETS: [FRACTOPO_WORKFLOW_VISUALISATION_PLOT],
    }


# Define default tasks
DOIT_CONFIG = {
    "default_tasks": [
        resolve_task_name(task_requirements),
        resolve_task_name(task_pre_commit),
        resolve_task_name(task_lint),
        # resolve_task_name(task_update_version),
        resolve_task_name(task_ci_test),
        resolve_task_name(task_docs),
        resolve_task_name(task_notebooks),
        resolve_task_name(task_build),
        resolve_task_name(task_codespell),
    ]
}

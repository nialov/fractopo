-   Use `git` to add and commit changes often. Assume you are working in
    a separate branch where the commits might often be rebased. Use
    `--no-gpg-sign` if you encounter errors with signing.
-   Use functional programming paradigms. Try to always write focused
    functions that do not have side effects.
-   Use `README.md` to understand the project.
-   Check existing project coding conventions from the code of the
    project.
-   Code structure: Main Python library code is in `./fractopo/`. Code
    for `marimo`-based web interface is in `./marimos/`. Documentation
    source is in `./docs_src/`. Code for `nix`, used for building and
    development, is in `./nix/`.
-   Instead of asking user, use the tools at your disposal to find the
    answer in the codebase.
-   Do not deviate from existing code structure.
-   Prefix commands with `nix develop -c` to run them in the development
    environment of `fractopo`. This applies to any commands listed in
    this `AGENTS.md` file.
-   Test Python code with `pytest`
-   Use `pre-commit` to lint and format code:
    `pre-commit run --all-files`

{ pkgs, ... }: {
  src = ./.;
  hooks = {
    nixfmt.enable = true;
    black.enable = true;
    flake8.enable = true;
    isort = {
      enable = true;
      raw = { args = [ "--profile" "black" ]; };
    };
    statix = { enable = true; };
    deadnix.enable = true;
    editorconfig-checker.enable = true;
    commitizen.enable = true;
    # update-changelog = {
    #   enable = true;
    #   name = "update-changelog";
    #   description = "update-changelog";
    #   entry = ''
    #     ${pkgs.update-changelog}/bin/update-changelog --changelog CHANGELOG.md
    #   '';
    #   # stages = [ "push" "manual" ];
    #   pass_filenames = false;
    # };
    sync-git-tag-with-poetry = {
      enable = true;
      name = "sync-git-tag-with-poetry";
      description = "sync-git-tag-with-poetry";
      entry = ''
        ${pkgs.sync-git-tag-with-poetry}/bin/sync-git-tag-with-poetry
      '';
      # stages = [ "push" "manual" ];
      pass_filenames = false;
    };
    trim-trailing-whitespace = {
      enable = true;

      name = "trim-trailing-whitespace";
      description = "This hook trims trailing whitespace.";
      entry =
        "${pkgs.python3Packages.pre-commit-hooks}/bin/trailing-whitespace-fixer";
      types = [ "text" ];
    };
    check-added-large-files = {
      enable = true;
      name = "check-added-large-files";
      description = "This hook checks for large added files.";
      entry =
        "${pkgs.python3Packages.pre-commit-hooks}/bin/check-added-large-files --maxkb=5000";
    };
    rstcheck = {
      enable = true;
      name = "rstcheck";
      description = "Check documentation with rstcheck";
      entry = "${pkgs.rstcheck}/bin/rstcheck";
      files = "\\.(rst)$";
      raw = { args = [ "-r" "docs_src" "--ignore-directives" "automodule" ]; };
    };
    cogapp = {
      enable = true;
      name = "cogapp";
      description = "Execute Python snippets in text files";
      entry = "${pkgs.python3Packages.cogapp}/bin/cog";
      files = "(README.rst|docs_src/index.rst)";
      pass_filenames = false;
      raw = {
        args = [ "-e" "-r" "--check" "-c" "docs_src/index.rst" ];
        always_run = true;
      };
    };
    # mypy-in-env = {
    #   enable = true;
    #   name = "mypy-in-env";
    #   description = "Run static type checks with mypy";
    #   entry = "${pkgs.poetryEnv}/bin/mypy";
    #   types = [ "python" ];
    #   stages = [ "manual" ];
    # };
    # pylint-in-env = {
    #   enable = true;
    #   name = "pylint-in-env";
    #   description = "Run pylint";
    #   entry = "${pkgs.poetryEnv}/bin/pylint";
    #   types = [ "python" ];
    #   stages = [ "manual" ];
    # };
  };
}

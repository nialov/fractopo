(
  { inputs, ... }:

  {
    perSystem =
      {
        self',
        config,
        system,
        pkgs,
        lib,
        ...
      }:
      let
        mkNixpkgs =
          nixpkgs:
          import nixpkgs {
            inherit system;
            overlays = [ inputs.self.overlays.default ];
            config = {
              allowUnfree = true;
            };
          };

      in
      {
        _module.args.pkgs = mkNixpkgs inputs.nixpkgs;
        devShells =
          let
            devShellPackages = with pkgs; [
              pre-commit
              fhs
              pythonEnv
              poetry
              ruff
            ];

          in
          {
            default = pkgs.mkShell {
              packages = devShellPackages;
              shellHook = config.pre-commit.installationScript + ''
                export PROJECT_DIR="$PWD"
                export PYTHONPATH="$PWD":"$PYTHONPATH"
              '';
            };

          };

        pre-commit = {
          check.enable = true;
          settings.hooks = {
            nixfmt.enable = true;
            nbstripout.enable = true;
            isort = {
              enable = true;
            };
            shellcheck = {enable = true;};
            statix.enable = true;
            deadnix.enable = true;
            rstcheck.enable = true;
            trim-trailing-whitespace.enable = true;
            check-added-large-files.enable = true;
            sync-git-tag-with-poetry.enable = false;
            editorconfig-checker.enable = false;
            cogapp = {
              enable = true;
              raw = {
                args = [ "docs_src/index.rst" ];
              };

            };
            yamllint = {
              enable = false;
              raw = {
                args = lib.mkBefore [
                  "-d"
                  "relaxed"
                ];
              };
            };
            commitizen.enable = true;
            ruff = {
              enable = true;
            };
            prettier = {
              enable = true;
              files = "\\.(geojson)$";
            };
          };

        };
        packages = {

          inherit (pkgs)
            fractopo
            poetry-run
            fractopo-network-run
            fractopo-validation-run
            push-fractopo-image
            ;
          fractopo-documentation = self'.packages.fractopo.passthru.documentation.doc;
          default = self'.packages.fractopo;
          fractopo-shell = self'.devShells.default;

        };
        checks = self'.packages;
        legacyPackages = pkgs;
      };

  }
)

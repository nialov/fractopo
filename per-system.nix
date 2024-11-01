({ self, inputs, ... }:

  {
    perSystem = { self', config, system, pkgs, lib, ... }:
      let
        mkNixpkgs = nixpkgs:
          import nixpkgs {
            inherit system;
            overlays = [

              inputs.nix-extra.overlays.default

              (final: prev: {
                pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
                  (_: pythonPrev: {
                    "fractopo" = pythonPrev.fractopo.overridePythonAttrs
                      # Test with local source
                      (_: { src = self.outPath; });
                  })
                ];
                inherit (final.python3Packages) fractopo;
              })

            ];
            config = { allowUnfree = true; };
          };

      in {
        _module.args.pkgs = mkNixpkgs inputs.nixpkgs;
        devShells = let
          devShellPackages = with pkgs; [
            pre-commit
            pandoc
            poetry-with-c-tooling
            # Supported python versions
            python39
            python310
            python311
          ];

        in {
          default = pkgs.mkShell {
            packages = devShellPackages;
            shellHook = config.pre-commit.installationScript + ''
              export PROJECT_DIR="$PWD"
            '';
          };

        };

        pre-commit = {
          check.enable = true;
          settings.hooks = {
            nixfmt.enable = true;
            black.enable = true;
            black-nb.enable = true;
            nbstripout.enable = true;
            isort = { enable = true; };
            shellcheck.enable = true;
            statix.enable = true;
            deadnix.enable = true;
            rstcheck.enable = true;
            trim-trailing-whitespace.enable = true;
            check-added-large-files.enable = true;
            sync-git-tag-with-poetry.enable = true;
            editorconfig-checker.enable = true;
            cogapp = {
              enable = true;
              raw = { args = [ "docs_src/index.rst" ]; };

            };
            yamllint = {
              enable = false;
              raw = { args = lib.mkBefore [ "-d" "relaxed" ]; };
            };
            commitizen.enable = true;
            ruff = { enable = true; };
            prettier = {
              enable = true;
              files = "\\.(geojson)$";
            };
          };

        };
        packages = {

          inherit (pkgs) fractopo poetry-run;
          fractopo310 = pkgs.python310Packages.fractopo;
          fractopo311 = pkgs.python311Packages.fractopo;
          fractopo-documentation =
            self'.packages.fractopo.passthru.documentation.doc;
          default = self'.packages.fractopo;

        };
        checks = self'.packages;
        legacyPackages = pkgs;
      };

  })

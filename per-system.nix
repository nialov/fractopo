({ inputs, ... }:

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
                  (pythonFinal: _: {
                    fractopo = pythonFinal.callPackage ./nix/package.nix { };
                  })
                ];
                inherit (final.python3Packages) fractopo;
                fhs = let
                  base = prev.geo-fhs-env.passthru.args;
                  config = {
                    name = "fhs";
                    targetPkgs = fhsPkgs:
                      (base.targetPkgs fhsPkgs) ++ [ fhsPkgs.gdal ];
                  };
                in pkgs.buildFHSUserEnv (lib.recursiveUpdate base config);

                pythonEnv = prev.python3.withPackages
                  (p: p.fractopo.passthru.optional-dependencies.dev);
              })

            ];
            config = { allowUnfree = true; };
          };

      in {
        _module.args.pkgs = mkNixpkgs inputs.nixpkgs;
        devShells =
          let devShellPackages = with pkgs; [ pre-commit fhs pythonEnv ];

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
            sync-git-tag-with-poetry.enable = false;
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
          fractopo-documentation =
            self'.packages.fractopo.passthru.documentation.doc;
          default = self'.packages.fractopo;
          fractopo-shell = self'.devShells.default;

        };
        checks = self'.packages;
        legacyPackages = pkgs;
      };

  })

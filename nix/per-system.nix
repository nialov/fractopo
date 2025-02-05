({ inputs, ... }:

  {
    perSystem = { self', config, system, pkgs, lib, ... }:
      let
        mkNixpkgs = nixpkgs:
          import nixpkgs {
            inherit system;
            overlays = [

              inputs.nix-extra.overlays.default

              (final: prev:

                let
                  imageConfig = {
                    name = "fractopo-validation";
                    config = {
                      Entrypoint = [
                        "${final.fractopo-validation-run}/bin/fractopo-validation-run"
                      ];
                      Cmd = [
                        "--host"
                        "0.0.0.0"
                        "--port"
                        "2718"
                        "--redirect-console-to-browser"
                      ];
                    };
                  };

                in {
                  pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
                    (pythonFinal: _: {
                      fractopo = pythonFinal.callPackage ./package.nix { };
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

                  pythonEnv = final.python3.withPackages
                    (p: p.fractopo.passthru.optional-dependencies.dev);
                  fractopoEnv = final.python3.withPackages (p:
                    # TODO: Should check.
                    [ p.fractopo.passthru.no-check ]
                    ++ p.fractopo.passthru.optional-dependencies.dev);
                  fractopo-validation-run = prev.writeShellApplication {
                    name = "fractopo-validation-run";
                    runtimeInputs = [ final.fractopoEnv ];
                    text = ''
                      marimo run ${./../marimos/validation.py} "$@"
                    '';

                  };
                  fractopo-validation-image =
                    pkgs.dockerTools.buildLayeredImage imageConfig;
                  fractopo-validation-image-stream =
                    pkgs.dockerTools.streamLayeredImage imageConfig;
                  push-fractopo-validation-image = prev.writeShellApplication {
                    name = "push-fractopo-validation-image";
                    text = let

                      inherit (final.fractopo-validation-image-stream)
                        imageName imageTag;

                    in ''
                      echo "Logging in to $1"
                      docker login -p "$3" -u unused "$1"

                      echo "Loading new version of fractopo validation image into docker"
                      ${final.fractopo-validation-image-stream} | docker load

                      echo "Listing images"
                      docker image list

                      echo "Tagging new image version to project $2 in $1"
                      docker tag ${imageName}:${imageTag} "$1"/"$2"/${imageName}:latest

                      echo "Pushing new image version to project $2 in $1"
                      docker push "$1"/"$2"/${imageName}:latest
                    '';
                  };
                  cut-release-changelog = prev.writeShellApplication {
                    name = "cut-release-changelog";
                    text = ''
                      ${prev.busybox}/bin/sed -n '3,/## v[[:digit:]].[[:digit:]].[[:digit:]]/p' CHANGELOG.md | head -n -2
                    '';
                  };
                })

            ];
            config = { allowUnfree = true; };
          };

      in {
        _module.args.pkgs = mkNixpkgs inputs.nixpkgs;
        devShells =
          let devShellPackages = with pkgs; [ pre-commit fhs pythonEnv poetry ];

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

          inherit (pkgs)
            fractopo poetry-run fractopo-validation-run
            fractopo-validation-image fractopo-validation-image-stream;
          fractopo-documentation =
            self'.packages.fractopo.passthru.documentation.doc;
          default = self'.packages.fractopo;
          fractopo-shell = self'.devShells.default;

        };
        checks = self'.packages;
        legacyPackages = pkgs;
      };

  })

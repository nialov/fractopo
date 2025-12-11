{ inputs, lib, ... }:

let

  inherit (inputs.actions-nix.lib.steps)
    actionsCheckout DeterminateSystemsNixInstallerAction runNixFastBuild
    runNixFlakeCheckNoBuild cachixCachixAction runBuildPackageWithPoetry
    runPublishPackageWithPoetry runCreateIncrementalChangelog
    softpropsActionGhRelease actionsUploadPagesArtifact actionsConfigurePages
    actionsDeployPages;
  inherit (inputs.actions-nix.lib.jobs)
    publishDocsToGitHubPages publishPackages;

  baseNixSteps = [
    actionsCheckout
    DeterminateSystemsNixInstallerAction
    {
      inherit (cachixCachixAction) uses;
      "with" = {
        name = "nialov";
        authToken = "\${{ secrets.CACHIX_AUTH_TOKEN }}";
      };
    }
  ];

in {
  flake.actions-nix = { config, ... }: {
    pre-commit.enable = true;
    defaults = {
      jobs = {
        timeout-minutes = 60;
        runs-on = "ubuntu-latest";
      };
    };
    workflows = {
      ".github/workflows/main.yaml" = {
        on = {
          push = { };
          workflow_dispatch = { };
        };
        jobs = {
          nix-fast-build = { steps = baseNixSteps ++ [ runNixFastBuild ]; };
          nix-flake-check-no-build = {
            steps = baseNixSteps ++ [ runNixFlakeCheckNoBuild ];
          };
          poetry = {
            strategy.matrix.python-version = [ "3.11" "3.12" "3.13" "3.14" ];
            steps = baseNixSteps ++ [{

              name = "Test with poetry on Python \${{ matrix.python-version }}";
              run =
                "nix run .#poetry-run -- \${{ matrix.python-version }} pytest";
            }];
          };
          release = {
            needs = [ "nix-fast-build" "nix-flake-check-no-build" "poetry" ];
            steps = let
              isTag = lib.concatStringsSep " && " [
                "github.event_name == 'push'"
                "startsWith(github.ref, 'refs/tags')"
              ];
            in baseNixSteps ++ [
              runBuildPackageWithPoetry

              (lib.recursiveUpdate runPublishPackageWithPoetry {
                "if" = isTag;
              })
              runCreateIncrementalChangelog
              (lib.recursiveUpdate softpropsActionGhRelease { "if" = isTag; })
            ];
          };
          docs = lib.recursiveUpdate publishDocsToGitHubPages {
            "if" = lib.concatStringsSep " && " [
              "github.event_name == 'push'"
              "startsWith(github.ref, 'refs/heads/master')"
            ];
            needs = [ "nix-fast-build" ];
            steps = baseNixSteps ++ [
              {
                name = "Build documentation";
                run = ''
                  nix build .#fractopo.passthru.documentation.doc
                  cp -Lr --no-preserve=mode,ownership,timestamps ./result-doc/share/doc/"$(nix eval --raw .#fractopo.name)"/html ./docs
                '';

              }
              (lib.recursiveUpdate actionsUploadPagesArtifact {
                "with".path = "docs/";
              })
              actionsConfigurePages
              actionsDeployPages
            ];
          };
          docker = lib.recursiveUpdate publishPackages {
            needs = [ "nix-fast-build" ];
            steps = let

              mkPushStep = { rev, name, ifConditions, }: {
                inherit name;
                "if" = lib.concatStringsSep " && " ifConditions;
                run = lib.concatStringsSep " " [
                  "nix run .#push-fractopo-image --"
                  ''ghcr.io nialov "$PUSHER_TOKEN"''
                  "\${{ github.actor }}"
                  rev
                ];
                env = { PUSHER_TOKEN = "\${{ secrets.GITHUB_TOKEN }}"; };
              };

            in baseNixSteps ++ [
              {
                name = "Load image to docker";
                run = "nix run .#load-fractopo-image";

              }
              (mkPushStep {
                rev = "$(git rev-parse --short HEAD)";
                name = "Push to ghcr.io from all branches";
                ifConditions = [ "github.event_name == 'push'" ];
              })
              (mkPushStep {
                rev = "latest";
                name = "Push to ghcr.io from default branch";
                ifConditions = [
                  "github.event_name == 'push'"
                  "github.ref == 'refs/heads/master'"
                ];
              })

            ];
          };
        };
      };
      ".github/workflows/conda.yaml" = {
        on = {
          push.paths = [
            "fractopo/**.py"
            "tests/**.py"
            ".github/workflows/conda.yaml"
            "pyproject.toml"
            "environment.yaml"
          ];
          workflow_dispatch = { };
        };
        jobs = {
          conda = {
            timeout-minutes = 30;
            strategy = {
              fail-fast = false;
              matrix = {
                inherit (config.workflows.".github/workflows/main.yaml".jobs.poetry.strategy.matrix)
                  python-version;
                platform = [ "ubuntu-latest" "macos-latest" "windows-latest" ];
              };
            };
            runs-on = "\${{ matrix.platform }}";
            defaults.run.shell = "bash -l {0}";
            steps = [

              actionsCheckout
              {
                uses = "mamba-org/setup-micromamba@v2";
                "with" = {
                  micromamba-version = "1.5.6-0";
                  environment-file = "environment.yaml";
                  init-shell = "bash powershell";
                  cache-environment = true;
                  cache-downloads = true;
                  post-cleanup = "all";
                  create-args = "python=\${{ matrix.python-version }}";
                };
              }
              {
                run = ''
                  echo "Testing package import"
                  python -c 'import fractopo'
                  echo "Testing module entrypoint"
                  python -m fractopo --help
                  echo "Running unittests with pytest"
                  pytest -v
                '';
              }

            ];
          };
        };
      };
    };
  };
}

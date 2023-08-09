{
  description = "nix declared development environment";

  inputs = {
    # nixpkgs.url = "nixpkgs/nixos-unstable";
    nixpkgs.url = "github:nixos/nixpkgs/b10a520";
    nix-extra = {
      url = "github:nialov/nix-extra";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # nixpkgs-fractopo.url =
    #   "github:NixOS/nixpkgs/a115bb9bd56831941be3776c8a94005867f316a7";
    # poetry2nix-copier.url =
    #   "github:nialov/poetry2nix?rev=6711fdb5da87574d250218c20bcd808949db6da0";
    flake-utils.url = "github:numtide/flake-utils";
    pre-commit-hooks = { url = "github:cachix/pre-commit-hooks.nix"; };
  };
  nixConfig.extra-substituters = [ "https://fractopo.cachix.org" ];
  nixConfig.extra-trusted-public-keys =
    [ "fractopo.cachix.org-1:Eo5bn5VTQSp4J3+XQnGYlq4dH/2ibKjxrs5n9qKl9Ms=" ];

  outputs = { self, nixpkgs, flake-utils, ... }@inputs:
    flake-utils.lib.eachSystem [ flake-utils.lib.system.x86_64-linux ] (system:
      let
        # Initialize nixpkgs for system
        pkgs = import nixpkgs {
          inherit system;
          overlays =
            [ self.overlays.default inputs.nix-extra.overlays.default ];
        };

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
        checks = {
          preCommitCheck = inputs.pre-commit-hooks.lib.${system}.run
            (import ././pre-commit.nix { inherit pkgs; });

        };
        packages = {
          inherit (pkgs)
            sync-git-tag-with-poetry resolve-version update-changelog
            pre-release poetry-run docs fractopo fractopo39 fractopo310
            fractopo311;
        };
        devShells = {
          default = pkgs.mkShell {
            packages = devShellPackages;
            inherit (self.checks.${system}.preCommitCheck) shellHook;
          };
          # poetry = self.packages."${system}".poetryEnv.env;
        };
      }) // {
        overlays.default = final: prev: {
          pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
            (python-final: _: {
              "fractopo" = python-final.callPackage ./default.nix { };
              # shapely = let
              #   shapelyPkgs =
              #     import inputs.nixpkgs-shapely { inherit (prev) system; };

              # in shapelyPkgs.python3Package.shapely;
            })
          ];

          # poetryEnv = prev.poetry2nix.mkPoetryEnv {
          #   projectDir = ./.;
          #   # editablePackageSources = { doit_ext = ./doit_ext; };
          # };
          # TODO: Does not seem to work?
          inherit (final.python3Packages) fractopo;
          fractopo39 = final.python39Packages.fractopo;
          fractopo310 = final.python310Packages.fractopo;
          fractopo311 = final.python311Packages.fractopo;
          docs = let
            sphinxEnv = final.python3.withPackages (p:
              with p; [
                sphinx
                sphinx-autodoc-typehints
                sphinx-rtd-theme
                sphinx-gallery
                nbsphinx
                matplotlib
                fractopo
                ipython
                notebook
              ]);
          in prev.runCommand "docs" {
            nativeBuildInputs = [ final.resolve-version prev.pandoc ];
          } ''
            tmpdir=$(mktemp -d)
            export HOME=$(mktemp -d)
            ln -s ${./. + "/fractopo"} $tmpdir/fractopo
            ln -s ${./README.rst} $tmpdir/README.rst
            cp -r ${./docs_src} $tmpdir/docs_src
            cp -r ${./examples} $tmpdir/examples
            mkdir -p $tmpdir/tests
            cp -r ${./tests/sample_data} $tmpdir/tests/sample_data
            chmod -R 777 $tmpdir/docs_src $tmpdir/examples
            cd $tmpdir
            ${sphinxEnv}/bin/sphinx-apidoc -o docs_src/apidoc -f fractopo -e -f
            ${sphinxEnv}/bin/sphinx-build -b html docs_src/ $out
          '';
          sync-git-tag-with-poetry = final.writeShellApplication {
            name = "sync-git-tag-with-poetry";
            runtimeInputs = with final; [ poetry git resolve-version ];
            text = ''
              version="$(resolve-version)"
              poetry version "$version"
            '';
          };
          resolve-version = prev.writeShellApplication {
            name = "resolve-version";
            runtimeInputs = with prev; [ git ];
            text = ''
              version="$(git tag --sort=-creatordate | head -n 1 | sed 's/v\(.*\)/\1/')"
              echo "$version"
            '';
          };
          update-changelog = prev.writeShellApplication {
            name = "update-changelog";
            runtimeInputs = with prev; [ clog-cli ripgrep pandoc ];
            text = ''
              homepage="$(rg 'homepage =' pyproject.toml | sed 's/.*"\(.*\)"/\1/')"
              version="$(git tag --sort=-creatordate | head -n 1 | sed 's/v\(.*\)/\1/')"
              clog --repository "$homepage" --subtitle "Release Changelog $version" "$@"
            '';
          };
          pre-release = final.writeShellApplication {
            name = "pre-release";
            runtimeInputs = with final; [
              update-changelog
              sync-git-tag-with-poetry
            ];
            text = ''
              sync-git-tag-with-poetry
              update-changelog --changelog CHANGELOG.md
              pandoc CHANGELOG.md --from markdown --to markdown --output CHANGELOG.md
            '';

          };
          poetry-run = prev.writeShellApplication {
            name = "poetry-run";
            runtimeInputs =
              self.devShells."${prev.system}".default.nativeBuildInputs;
            text = ''
              poetry check
              poetry env use "$1"
              shift
              poetry env info
              poetry lock --check
              poetry install
              poetry run "$@"
            '';

          };

        };
      };

}

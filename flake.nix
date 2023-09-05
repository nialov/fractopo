{
  description = "fractopo: A Python package for fracture network analysis";

  inputs = {
    # nixpkgs.url = "nixpkgs/nixos-unstable";
    # TODO: Must use nixpkgs with old shapely version until shapely 2.0 support
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nix-extra = {
      url = "github:nialov/nix-extra";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
    pre-commit-hooks = { url = "github:cachix/pre-commit-hooks.nix"; };
  };
  nixConfig.extra-substituters = [ "https://fractopo.cachix.org" ];
  nixConfig.extra-trusted-public-keys =
    [ "fractopo.cachix.org-1:Eo5bn5VTQSp4J3+XQnGYlq4dH/2ibKjxrs5n9qKl9Ms=" ];

  outputs = { self, nixpkgs, flake-utils, ... }@inputs:
    let inherit (nixpkgs) lib;
    in lib.recursiveUpdate
    (flake-utils.lib.eachSystem [ flake-utils.lib.system.x86_64-linux ] (system:
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
        checks = lib.recursiveUpdate {
          preCommitCheck = inputs.pre-commit-hooks.lib.${system}.run
            (import ././pre-commit.nix { inherit pkgs; });

        } self.packages."${system}";
        packages = {
          inherit (pkgs)
            sync-git-tag-with-poetry resolve-version update-changelog
            pre-release fractopo;
          poetry-run = pkgs.poetry-run-fractopo;
          # django not supported for python 3.9
          # fractopo39 = pkgs.python39Packages.fractopo;
          fractopo310 = pkgs.python310Packages.fractopo;
          # TODO: 311 fails due to python3.11-twisted-22.10.0.drv' failed with exit code 1;
          # fractopo311 = pkgs.python311Packages.fractopo;
        };
        devShells = {
          default = pkgs.mkShell {
            packages = devShellPackages;
            inherit (self.checks.${system}.preCommitCheck) shellHook;
          };
        };
      })) {
        overlays.default = final: prev: {
          pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
            (python-final: _: {
              "fractopo" = python-final.callPackage ./default.nix { };
            })
          ];

          inherit (final.python3Packages) fractopo;
          poetry-run-fractopo = final.poetry-run.override {
            pythons = with prev; [ python39 python310 python311 ];
          };

        };
      };

}

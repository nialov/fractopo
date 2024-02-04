{
  description = "fractopo: A Python package for fracture network analysis";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nix-extra = {
      url = "github:nialov/nix-extra";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-parts.follows = "nix-extra/flake-parts";
    flake-utils.follows = "nix-extra/flake-utils";
    pre-commit-hooks.follows = "nix-extra/pre-commit-hooks";
  };
  nixConfig.extra-substituters = [ "https://fractopo.cachix.org" ];
  nixConfig.extra-trusted-public-keys =
    [ "fractopo.cachix.org-1:Eo5bn5VTQSp4J3+XQnGYlq4dH/2ibKjxrs5n9qKl9Ms=" ];

  outputs = { self, nixpkgs, ... }@inputs:
    let
      flakePart = inputs.flake-parts.lib.mkFlake { inherit inputs; }
        ({ inputs, ... }: {
          systems = [ "x86_64-linux" ];
          imports = [
            inputs.nix-extra.flakeModules.custom-pre-commit-hooks
            inputs.nix-extra.flakeModules.poetryDevshell
          ];
          flake = {
            overlays.default = inputs.nixpkgs.lib.composeManyExtensions [
              inputs.nix-extra.overlays.utils
              (final: prev: {
                pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
                  (python-final: _: {
                    "fractopo" = python-final.callPackage ./default.nix { };
                  })
                ];
                inherit (final.python3Packages) fractopo;
                poetry-run-fractopo = final.poetry-run.override {
                  pythons = with prev; [
                    python38
                    python39
                    python310
                    python311
                  ];
                };
              })
            ];
          };
          perSystem = { config, system, pkgs, lib, self', ... }:
            let
              mkNixpkgs = nixpkgs:
                import nixpkgs {
                  inherit system;
                  overlays = [ self.overlays.default ];
                  config = { allowUnfree = true; };
                };

            in {
              _module.args.pkgs = mkNixpkgs inputs.nixpkgs;
              checks = self'.packages;
              packages = {
                inherit (pkgs) fractopo;
                poetry-run = pkgs.poetry-run-fractopo;
                # django not supported for python <3.10
                # fractopo38 = pkgs.python38Packages.fractopo;
                # fractopo39 = pkgs.python39Packages.fractopo;
                fractopo310 = pkgs.python310Packages.fractopo;
                fractopo311 = pkgs.python311Packages.fractopo;
                fractopo-documentation =
                  pkgs.fractopo.passthru.documentation.doc;
                default = self'.packages.fractopo;
              };
              devShells.default = self'.devShells.poetry-devshell.overrideAttrs
                (prevAttrs: {
                  buildInputs = prevAttrs.buildInputs
                    ++ [ pkgs.poetry-run-fractopo ];
                });
              pre-commit = {
                check.enable = true;
                settings.hooks = {

                  nixfmt.enable = true;
                  black.enable = true;
                  flake8.enable = true;
                  isort = { enable = true; };
                  statix = { enable = true; };
                  deadnix.enable = true;
                  editorconfig-checker.enable = true;
                  commitizen.enable = true;
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
                  trim-trailing-whitespace = { enable = true; };
                  check-added-large-files = { enable = true; };
                  rstcheck = {
                    enable = true;
                    raw = { args = [ "-r" "docs_src" ]; };
                  };
                  cogapp = {
                    enable = true;
                    files = "(README.rst|docs_src/index.rst)";
                    pass_filenames = false;
                    raw = {
                      args = lib.mkAfter [ "docs_src/index.rst" ];
                      always_run = true;
                    };
                  };
                  nbstripout = { enable = true; };
                };
              };
            };

        });
    in flakePart;

}

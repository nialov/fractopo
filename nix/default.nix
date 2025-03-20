inputs:
let
  flakePart = inputs.flake-parts.lib.mkFlake { inherit inputs; }
    ({ self, inputs, ... }: {
      systems = [ "x86_64-linux" ];
      imports = [
        inputs.nix-extra.flakeModules.custom-pre-commit-hooks
        inputs.actions-nix.flakeModules.default
        ./per-system.nix
        ./ci.nix
      ];
      flake = { inherit self; };
    });

in flakePart

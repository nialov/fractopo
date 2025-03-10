inputs:
let
  flakePart = inputs.flake-parts.lib.mkFlake { inherit inputs; }
    ({ self, inputs, ... }: {
      systems = [ "x86_64-linux" ];
      imports = [
        inputs.nix-extra.flakeModules.custom-pre-commit-hooks
        ./per-system.nix
      ];
      flake = { inherit self; };
    });

in flakePart

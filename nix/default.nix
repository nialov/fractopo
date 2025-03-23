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
      flake = {
        inherit self;
        overlays = let overlays' = import ./overlays.nix;
        in {
          inherit (overlays') packageOverlay localOverlay;
          default = inputs.nixpkgs.lib.composeManyExtensions [
            inputs.nix-extra.overlays.default
            self.overlays.packageOverlay
            self.overlays.localOverlay
          ];
        };
      };
    });

in flakePart

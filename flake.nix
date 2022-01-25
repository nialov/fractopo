{
  description = "nix declared development environment";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-21.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
      (flake-utils.lib.eachDefaultSystem (system:
        let pkgs = nixpkgs.legacyPackages."${system}"; in
        {
          devShell = import ././shell.nix {
            pkgs=pkgs;
          };
        }
      ));
}  

{
  description = "Description for the project";

  inputs = {
    nix-extra = {
      url = "github:nialov/nix-extra";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-parts.follows = "nix-extra/flake-parts";
    actions-nix = {
      url = "github:nialov/actions.nix";
      inputs.git-hooks.follows = "nix-extra/git-hooks";
      inputs.flake-parts.follows = "nix-extra/flake-parts";
    };
  };

  outputs = inputs: (import ./nix) inputs;

}

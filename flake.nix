{
  description = "Description for the project";

  inputs = {
    nix-extra = { url = "github:nialov/nix-extra"; };
    nixpkgs.follows = "nix-extra/nixpkgs";
    flake-parts.follows = "nix-extra/flake-parts";
    actions-nix = { url = "github:nialov/actions.nix"; };
  };

  outputs = inputs: (import ./nix) inputs;

}

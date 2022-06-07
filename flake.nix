{
  description = "nix declared development environment";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    let
      mkshell = pkgs:
        let
          poetry-wrapped = pkgs.callPackage ({ writeScriptBin, poetry, stdenv
            , zlib, cacert, python37, python38, python39, lib, execline }:
            let
              pythons = [ python37 python38 python39 ];

              site-packages = lib.concatStringsSep ":" (lib.forEach pythons
                (python: "${python}/${python.sitePackages}"));
              interpreters = lib.concatStringsSep ":"
                (lib.forEach pythons (python: "${python}/bin"));
            in writeScriptBin "poetry" ''
              CLIB="${stdenv.cc.cc.lib}/lib"
              ZLIB="${zlib}/lib"
              CERT="${cacert}/etc/ssl/certs/ca-bundle.crt"

              export GIT_SSL_CAINFO=$CERT
              export SSL_CERT_FILE=$CERT
              export CURL_CA_BUNDLE=$CERT
              export LD_LIBRARY_PATH=$CLIB:$ZLIB

              export PYTHONPATH=${site-packages}
              export PATH=${interpreters}:$PATH
              ${execline}/bin/exec -a "$0" "${poetry}/bin/poetry" "$@"
            '') { };
        in pkgs.mkShell rec {
          packages = with pkgs; [
            pre-commit
            pandoc
            git
            pastel
            nixFlakes
            poetry-wrapped
          ];

          envrc_contents = ''
            use flake
          '';

          shellHook = ''
            [[ -a .pre-commit-config.yaml ]] && \
              echo "Installing pre-commit hooks"; pre-commit install
            pastel paint -n green "
            Run poetry install to install environment from poetry.lock
            "
            [[ ! -a .envrc ]] && echo -n "$envrc_contents" > .envrc
          '';
        };
    in flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages."${system}";
      in { devShell = mkshell pkgs; });
}

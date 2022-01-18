{ pkgs }:

let
  dev-pkgs = python-packages: with python-packages; [
    pipx
  ];
  python38Pipx = pkgs.python38.withPackages dev-pkgs;
in

pkgs.mkShell {
  buildInputs = with pkgs; [
    poetry
    python37
    python38
    python39
    pre-commit
    pandoc
    git
    cacert
  ];

  shellHook = with pkgs; ''
    echo Setting environment for shared libraries
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${stdenv.cc.cc.lib}/lib
    export GIT_SSL_CAINFO=${cacert}/etc/ssl/certs/ca-bundle.crt
    export SSL_CERT_FILE=${cacert}/etc/ssl/certs/ca-bundle.crt
    export CURL_CA_BUNDLE=${cacert}/etc/ssl/certs/ca-bundle.crt
    [[ -a .pre-commit-config.yaml ]] && \
      echo "Installing pre-commit hooks"; pre-commit install
    export PYTHONPATH=${python37}/lib/python3.7/site-packages
    export PYTHONPATH=${python38}/lib/python3.8/site-packages
    export PYTHONPATH=${python39}/lib/python3.9/site-packages
    echo Run poetry install to install environment from poetry.lock
  '';
}

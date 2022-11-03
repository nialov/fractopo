{
  description = "nix declared development environment";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    let
      # Create function to generate the poetry-included shell with single
      # input: pkgs
      poetry-wrapped-generate = { pkgs, pythons, poetry ? pkgs.poetry }:
        let
          inherit (pkgs) lib;
          # The wanted python interpreters are set here. E.g. if you want to
          # add Python 3.7, add 'python37'.
          # pythons = with pkgs; [ python38 python39 python310 ];

          # The paths to site-packages are extracted and joined with a colon
          site-packages = lib.concatStringsSep ":"
            (lib.forEach pythons (python: "${python}/${python.sitePackages}"));

          # The paths to interpreters are extracted and joined with a colon
          interpreters = lib.concatStringsSep ":"
            (lib.forEach pythons (python: "${python}/bin"));

          # Create a script with the filename poetry so that all "poetry"
          # prefixed commands run the same. E.g. you can use 'poetry run'
          # normally. The script sets environment variables before passing
          # all arguments to the poetry executable These environment
          # variables are required for building Python packages with e.g. C
          # -extensions.
        in pkgs.writeScriptBin "poetry" ''
          CLIB="${pkgs.stdenv.cc.cc.lib}/lib"
          ZLIB="${pkgs.zlib}/lib"
          CERT="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"

          export GIT_SSL_CAINFO=$CERT
          export SSL_CERT_FILE=$CERT
          export CURL_CA_BUNDLE=$CERT
          export LD_LIBRARY_PATH=$CLIB:$ZLIB

          export PYTHONPATH=${site-packages}
          export PATH=${interpreters}:$PATH
          ${pkgs.execline}/bin/exec -a "$0" "${poetry}/bin/poetry" "$@"
        '';
      # Define the actual development shell that contains the now wrapped
      # poetry executable 'poetry-wrapped'
      # mkshell = pkgs:

      # Use flake-utils to declare the development shell for each system nix
      # supports e.g. x86_64-linux and x86_64-darwin (but no guarantees are
      # given that it works except for x86_64-linux, which I use).
    in flake-utils.lib.eachSystem [ flake-utils.lib.system.x86_64-linux ]
    (system:
      let
        pkgs = nixpkgs.legacyPackages."${system}";
        # Pass pkgs input to poetry-wrapped-generate function which then
        # returns the poetry-wrapped package.
        # poetry-wrapped = poetry-wrapped-generate {
        #   inherit pkgs;
        #   pythons = with pkgs; [ python39 ];
        #   poetry = pkgs.python39Packages.poetry;
        # };
      in {
        devShells = let
          mkPoetryShell = pythonv:
            let
              poetry-wrapped = poetry-wrapped-generate {
                inherit pkgs;
                pythons = [ pythonv ];
              };
            in pkgs.mkShell {
              # The development environment can contain any tools from nixpkgs
              # alongside poetry Here we add e.g. pre-commit and pandoc
              packages = with pkgs; [
                pre-commit
                pandoc
                poetry-wrapped
                git
                # (doit.overrideAttrs (prev: {
                #   propagatedBuildInputs = prev.propagatedBuildInputs
                #     ++ [ python3Packages.tomli ];
                # }))
              ];

              envrc_contents = ''
                use flake
              '';

              # Define a shellHook that is called every time that development shell
              # is entered. It installs pre-commit hooks and prints a message about
              # how to install python dependencies with poetry. Lastly, it
              # generates an '.envrc' file for use with 'direnv' which I recommend
              # using for easy usage of the development shell
              shellHook = ''
                [[ -a .pre-commit-config.yaml ]] && \
                  echo "Installing pre-commit hooks"; pre-commit install
                [[ ! -a .envrc ]] && echo -n "$envrc_contents" > .envrc
                ${poetry-wrapped}/bin/poetry env use ${pythonv.interpreter}
                ${pkgs.pastel}/bin/pastel paint -n green "
                Run poetry install to install environment from poetry.lock
                "
              '';
            };
          checks = let
            mkCheck = pythonv:
              let
                poetry-wrapped = poetry-wrapped-generate {
                  inherit pkgs;
                  pythons = [ pythonv ];
                };
              in {
                test-poetry-wrapped =
                  pkgs.runCommand "test-poetry-wrapped" { } ''
                    ${poetry-wrapped}/bin/poetry --help
                    ${poetry-wrapped}/bin/poetry init -n
                    ${poetry-wrapped}/bin/poetry check
                    mkdir $out
                  '';
              };
          in {
            python38 = mkCheck pkgs.python38;
            python39 = mkCheck pkgs.python39;
            python310 = mkCheck pkgs.python310;

          };
        in {
          python38 = mkPoetryShell pkgs.python38;
          python39 = mkPoetryShell pkgs.python39;
          python310 = mkPoetryShell pkgs.python310;
          default = self.devShells."${system}".python39;
        };
        packages = let
          genImg = pythonv:
            let
              poetry-wrapped = poetry-wrapped-generate {
                inherit pkgs;
                # Do not need to specify poetry version as interpreter is explicitly set
                inherit (pkgs) poetry;
                # TODO: No need for multiple pythons
                pythons = [ pythonv ];
              };
              script = pkgs.writeShellScriptBin "test-script" ''
                ${poetry-wrapped}/bin/poetry check && \
                    ${poetry-wrapped}/bin/poetry env use ${pythonv.interpreter} && \
                    ${poetry-wrapped}/bin/poetry install && \
                    ${poetry-wrapped}/bin/poetry run pytest --collect-only
              '';
            in pkgs.dockerTools.streamLayeredImage {
              name = "layered-image";
              tag = "latest";
              extraCommands = ''
                cp -r ${./.} ./src/
                chmod 777 ./src
              '';
              config = {
                Cmd = [ "${script}/bin/test-script" ];
                WorkingDir = "/src";
                Env = [ "POETRY_CACHE_DIR=/poetry-cache" ];
              };
              # contents = with pkgs; [ pkgs.hello pkgs.bash pkgs.coreutils curl ];
            };
        in {
          # python37-fractopo-test = genImg pkgs.python37;
          python38-fractopo-test = genImg pkgs.python38;
          python39-fractopo-test = genImg pkgs.python39;
          python310-fractopo-test = genImg pkgs.python310;
        };
      });
}

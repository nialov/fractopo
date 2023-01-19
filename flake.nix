{
  description = "nix declared development environment";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    nixpkgs-copier.url =
      "github:nialov/nixpkgs?rev=334c000bbbc51894a3b02e05375eae36ac03e137";
    nixpkgs-old-shapely.url =
      "github:nixos/nixpkgs?rev=2b6c72e17306cb57b97ab1e26bb4c7a42ba1e430";
    poetry2nix-copier.url =
      "github:nialov/poetry2nix?rev=6711fdb5da87574d250218c20bcd808949db6da0";
    flake-utils.url = "github:numtide/flake-utils";
    copier-src = {
      url =
        # "github:copier-org/copier/precommix?rev=51a5bd9878ad036c69ef7e4ae8b0f313bdf180ec";
        "github:copier-org/copier?rev=51a5bd9878ad036c69ef7e4ae8b0f313bdf180ec";
      # TODO: copier flake requires specific nixpkgs entries
      # See: https://github.com/copier-org/copier/blob/51a5bd9878ad036c69ef7e4ae8b0f313bdf180ec/flake.nix
      inputs.nixpkgs.follows = "nixpkgs-copier";
      inputs.poetry2nix.follows = "poetry2nix-copier";
    };
  };
  nixConfig.extra-substituters = [ "https://fractopo.cachix.org" ];
  nixConfig.extra-trusted-public-keys =
    [ "fractopo.cachix.org-1:Eo5bn5VTQSp4J3+XQnGYlq4dH/2ibKjxrs5n9qKl9Ms=" ];

  outputs = { self, nixpkgs, flake-utils, copier-src, ... }@inputs:
    let
      # Create function to generate the poetry-included shell with single
      # input: pkgs
      wrapPoetry = { pkgs, pythons }:
        let
          inherit (pkgs) lib;
          # The wanted python interpreters are set here. E.g. if you want to
          # add Python 3.7, add 'python37'.
          pythonPkgs = lib.forEach pythons (python: pkgs."${python}");
          # inherit pythons;

          # The paths to site-packages are extracted and joined with a colon.
          sitePackages = lib.concatStringsSep ":" (lib.forEach pythonPkgs
            (python: "${python}/${python.sitePackages}"));

          # The paths to interpreters are extracted and joined with a colon.
          interpreters = lib.concatStringsSep ":"
            (lib.forEach pythonPkgs (python: "${python}/bin"));

          # Use latest poetry version from nixpkgs.
          # The poetry dev shell might use another Python interpreter.
          # That is set explicitly in the shellHook.
          inherit (pkgs) poetry;

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

          export PYTHONPATH=${sitePackages}
          export PATH=${interpreters}:$PATH
          ${pkgs.execline}/bin/exec -a "$0" "${poetry}/bin/poetry" "$@"
        '';
      # Define the actual development shell that contains the now wrapped
      # poetry executable 'poetry-wrapped'
      makeDevShellWithPoetry =
        { pkgs, pythons, defaultPython, devShellPackages }:
        pkgs.mkShell {
          # The development environment can contain any tools from nixpkgs
          # alongside poetry Here we add e.g. pre-commit and pandoc
          # packages = with pkgs; [ pre-commit pandoc wrappedPoetry ];
          packages = devShellPackages;

          envrc_contents = ''
            use flake
          '';

          # Define a shellHook that is called every time that development shell
          # is entered. It installs pre-commit hooks and prints a message about
          # how to install python dependencies with poetry. Lastly, it
          # generates an '.envrc' file for use with 'direnv' which I recommend
          # using for easy usage of the development shell
          shellHook = let

            defaultPythonPkg = pkgs.${defaultPython};
            # Install pre-commit hooks
            installPrecommit = ''
              export PRE_COMMIT_HOME=$(pwd)/.pre-commit-cache
              [[ -a .pre-commit-config.yaml ]] && \
                echo "Installing pre-commit hooks"; pre-commit install '';
            # Report how to install poetry packages
            reportPoetry = ''
              ${pkgs.pastel}/bin/pastel paint -n green "
              Run poetry install to install environment from poetry.lock
              "
            '';
            # Generate .envrc if it does not exist
            createEnvrc = ''
              [[ ! -a .envrc ]] && echo -n "$envrc_contents" > .envrc
            '';
            # Set poetry to use specific Python interpreter
            setPoetryEnv = ''
              poetry env use ${defaultPythonPkg.interpreter}
            '';
          in ''
            ${installPrecommit}
            ${setPoetryEnv}
            ${reportPoetry}
            ${createEnvrc}
          '';
        };
      # Use flake-utils to declare the development shell for each system nix
      # supports e.g. x86_64-linux and x86_64-darwin (but no guarantees are
      # given that it works except for x86_64-linux, which I use).
    in flake-utils.lib.eachSystem [ flake-utils.lib.system.x86_64-linux ]
    (system:
      let
        fractopoOverlay = final: prev:
          let
            overridePython = python:
              let
                self = prev."${python}".override {
                  inherit self;
                  packageOverrides =
                    prev.lib.composeManyExtensions final.pythonPackagesOverlays;
                };
              in self;
          in {
            # TODO: Fails on python38 with nix build
            pythonPackagesOverlays = (prev.pythonPackagesOverlays or [ ]) ++ [
              (python-final: python-prev: {
                fractopo = python-final.callPackage ./. { };
                # TODO: Remove when nixpkgs gets updated upstream with fix
                pygeos = python-prev.pygeos.overrideAttrs
                  (finalAttrs: prevAttrs: { patches = [ ]; });
                # shapely =
                #   inputs.nixpkgs-old-shapely.legacyPackages."${system}".python3Packages.shapely;
                # TODO: Only required as long as shapely 2.0 is not supported
                inherit (inputs.nixpkgs-old-shapely.legacyPackages."${system}".python3Packages)
                  shapely;
              })
            ];
            python3 = overridePython "python3";
            python39 = overridePython "python39";
            python310 = overridePython "python310";
            # python311 = overridePython "python311";

            python3Packages = final.python3.pkgs;
            python39Packages = final.python39.pkgs;
            python310Packages = final.python310.pkgs;
            # python311Packages = final.python311.pkgs;
          };
        # Initialize nixpkgs for system
        pkgs = import nixpkgs {
          inherit system;
          # Add copier overlay to provide copier package
          overlays = [
            fractopoOverlay
            (_: prev: { copier = copier-src.packages."${system}".default; })
          ];
        };

        # Choose Python interpreters to include in all devShells
        pythons = [ "python38" "python39" "python310" "python311" ];

        # Choose default Python interpreter to use with poetry
        defaultPython = "python38";

        # wrappedPoetry is also included as a flake output package
        wrappedPoetry = wrapPoetry { inherit pkgs pythons; };

        # copier (copier --help) does not work without git in its PATH
        wrappedCopier = pkgs.symlinkJoin {
          name = "copier";
          paths = [ pkgs.copier ];
          buildInputs = [ pkgs.makeWrapper ];
          postBuild = let gitPath = with pkgs; lib.makeBinPath [ git ];
          in ''
            wrapProgram $out/bin/copier \
              --prefix PATH : ${gitPath}
          '';
        };

        # Any packages from nixpkgs can be added here
        devShellPackages = with pkgs;
          [ pre-commit pandoc wrappedPoetry wrappedCopier ]
          ++ lib.forEach pythons (p: pkgs."${p}");

        # Generate devShells for wanted Pythons
        devShells = builtins.foldl' (x: y: (pkgs.lib.recursiveUpdate x y)) { }
          (pkgs.lib.forEach pythons (python: {
            "${python}" = makeDevShellWithPoetry {
              inherit pkgs pythons devShellPackages;
              defaultPython = python;
            };
          }));

        # Add default devShell
        devShellsWithDefault = pkgs.lib.recursiveUpdate devShells {
          default = devShells."${defaultPython}";
        };
        wrappedPoetryCheck =
          # let
          #   mkCheck = python:
          let wrappedPoetry = wrapPoetry { inherit pkgs pythons; };
          in pkgs.runCommand "test-poetry-wrapped" { } ''
            ${wrappedPoetry}/bin/poetry --help
            ${wrappedPoetry}/bin/poetry init -n
            ${wrappedPoetry}/bin/poetry check
            mkdir $out
          '';
        copierCheck = pkgs.runCommand "test-copier" { } ''
          ${wrappedCopier}/bin/copier --help
          mkdir $out
        '';
        fractopo-image = let inherit (self.packages."${system}") fractopo;
        in pkgs.dockerTools.buildLayeredImage {
          name = "fractopo";
          tag = "latest";
          config.Cmd = [ "${fractopo}/bin/fractopo" ];
          contents = [ fractopo ];

        };
      in {
        checks = { inherit wrappedPoetryCheck copierCheck; };
        packages.poetry-wrapped = wrappedPoetry;
        packages.copier = wrappedCopier;
        packages.fractopo = pkgs.python3Packages.fractopo;
        # packages.fractopo-38 = pkgs.fractopo-38;
        packages.fractopo-39 = pkgs.python39Packages.fractopo;
        packages.fractopo-310 = pkgs.python310Packages.fractopo;
        # packages.fractopo-311 = pkgs.python311Packages.fractopo;
        packages.fractopo-image = fractopo-image;
        packages.default = self.packages.fractopo;
        devShells = devShellsWithDefault;
        overlays.default = fractopoOverlay;
      });
}

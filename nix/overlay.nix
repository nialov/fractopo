(final: prev:

  let
    inherit (prev) lib;
    mkImageConfig = { name, entrypoint }: {
      inherit name;
      extraCommands = ''
        mkdir ./app
        mkdir -p ./tmp
        chmod 777 ./app
        chmod 777 ./tmp
      '';
      # Add for debugging
      contents = [ prev.bashInteractive prev.busybox ];
      config = {
        Entrypoint = [ entrypoint ];
        WorkingDir = "/app";
        Env = [ "HOME=/app" ];
        Cmd = [
          "--host"
          "0.0.0.0"
          "--port"
          "2718"
          "--redirect-console-to-browser"
        ];
      };
    };
    validationImageConfig = mkImageConfig {
      name = "fractopo-validation";
      entrypoint =
        "${final.fractopo-validation-run}/bin/fractopo-validation-run";
    };
    networkImageConfig = mkImageConfig {
      name = "fractopo-network";
      entrypoint = "${final.fractopo-network-run}/bin/fractopo-network-run";
    };
    mkMarimoRun = { name, script, marimosDir ? ../marimos }:
      prev.writeShellApplication {
        inherit name;
        runtimeInputs = [ final.fractopoEnv ];
        text = let scriptPath = "${marimosDir}/${script}";
        in ''
          marimo run ${scriptPath} "$@"
        '';

      };

  in {
    pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
      (pythonFinal: _: {
        fractopo = pythonFinal.callPackage ./package.nix { };
      })
    ];
    inherit (final.python3Packages) fractopo;
    fhs = let
      base = prev.geo-fhs-env.passthru.args;
      config = {
        name = "fhs";
        targetPkgs = fhsPkgs: (base.targetPkgs fhsPkgs) ++ [ fhsPkgs.gdal ];
      };
    in prev.buildFHSUserEnv (lib.recursiveUpdate base config);

    pythonEnv = final.python3.withPackages
      (p: p.fractopo.passthru.optional-dependencies.dev);
    fractopoEnv = final.python3.withPackages (p:
      # TODO: Should check.
      [ p.fractopo.passthru.no-check ]
      ++ p.fractopo.passthru.optional-dependencies.dev);
    fractopo-validation-run = mkMarimoRun {
      name = "fractopo-validation-run";
      script = "validation.py";
    };

    fractopo-network-run = mkMarimoRun {
      name = "fractopo-network-run";
      script = "network.py";

    };

    fractopo-validation-image =
      prev.dockerTools.buildLayeredImage validationImageConfig;
    fractopo-validation-image-stream =
      prev.dockerTools.streamLayeredImage validationImageConfig;

    fractopo-network-image =
      prev.dockerTools.buildLayeredImage networkImageConfig;
    fractopo-network-image-stream =
      prev.dockerTools.streamLayeredImage networkImageConfig;
    push-fractopo-images = prev.writeShellApplication {
      name = "push-fractopo-images";
      text = let

        streams = [
          final.fractopo-validation-image-stream
          final.fractopo-network-image-stream
        ];

        mkLoadCmd = stream: "${stream} | docker load";
        loadCmds = builtins.map mkLoadCmd streams;

        mkTagCmd = { imageName, imageTag }:
          ''docker tag ${imageName}:${imageTag} "$1"/"$2"/${imageName}:latest'';

        tagCmds = builtins.map
          (stream: mkTagCmd { inherit (stream) imageName imageTag; }) streams;

        mkPushCmd = imageName: ''docker push "$1"/"$2"/${imageName}:latest'';

        pushCmds = builtins.map (stream: mkPushCmd stream.imageName) streams;

      in ''
        echo "Logging in to $1"
        docker login -p "$3" -u unused "$1"

        echo "Loading new version of fractopo images into docker"
        ${lib.concatStringsSep "\n" loadCmds}

        echo "Listing images"
        docker image list

        echo "Tagging new image versions to project $2 in $1"
        ${lib.concatStringsSep "\n" tagCmds}

        echo "Pushing new image versions to project $2 in $1"
        ${lib.concatStringsSep "\n" pushCmds}
      '';
    };
    cut-release-changelog = prev.writeShellApplication {
      name = "cut-release-changelog";
      text = ''
        ${prev.busybox}/bin/sed -n '3,/## v[[:digit:]].[[:digit:]].[[:digit:]]/p' CHANGELOG.md | head -n -2
      '';
    };
  })

(

  let

    packageOverlay = _final: prev: {
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (pythonFinal: _: {
          fractopo = pythonFinal.callPackage ./package.nix { };
        })
      ];
    };

    localOverlay = final: prev:
      let
        inherit (prev) lib;
        fractopoImageConfig = {
          name = "fractopo-app";
          extraCommands = ''
            mkdir -p ./tmp
            chmod -R 777 ./tmp
            mkdir -p ./app/marimos
            cp ${../marimos/utils.py} ./app/marimos/utils.py
            cp ${../marimos/validation.py} ./app/marimos/validation.py
            cp ${../marimos/network.py} ./app/marimos/network.py
            chmod -R 777 ./app
          '';
          # Add for debugging
          contents = [ prev.bashInteractive prev.busybox ];
          config = {
            Entrypoint = [
              "/bin/bash"
              "-c"
              (lib.concatStringsSep " " [
                "${final.fractopoEnv}/bin/marimo"
                "run"
                "--host"
                "$HOST"
                "--port"
                "$PORT"
                "--redirect-console-to-browser"
                "/app/marimos/$RUN_MODE.py"
              ])
            ];
            WorkingDir = "/app";
            Env = [ "HOME=/app" "HOST=0.0.0.0" "PORT=2718" "RUN_MODE=network" ];
          };
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
        inherit (final.python3Packages) fractopo;
        fractopo-fhs = let
          base = prev.fhs.passthru.args;
          config = {
            name = "fhs";
            targetPkgs = fhsPkgs:
              (base.targetPkgs fhsPkgs)
              ++ [ fhsPkgs.gdal fhsPkgs.stdenv.cc.cc.lib ];
            profile = ''
              export LD_LIBRARY_PATH=${prev.stdenv.cc.cc.lib}/lib/
            '';
          };
        in prev.buildFHSEnv (lib.recursiveUpdate base config);

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

        fractopo-app-image =
          prev.dockerTools.buildLayeredImage fractopoImageConfig;
        fractopo-app-image-stream =
          prev.dockerTools.streamLayeredImage fractopoImageConfig;

        load-fractopo-image = prev.writeShellApplication {
          name = "load-fractopo-image";
          text = ''
            echo "Loading new version of fractopo image into docker"
            ${final.fractopo-app-image-stream} | docker load

            echo "Listing images"
            docker image list
          '';
        };
        push-fractopo-image = prev.writeShellApplication {
          name = "push-fractopo-image";
          text = let

            mkTagCmd = { imageName, imageTag }:
              ''
                docker tag ${imageName}:${imageTag} "$1"/"$2"/${imageName}:"$5"'';

            tagCmd = mkTagCmd {
              inherit (final.fractopo-app-image-stream) imageName imageTag;
            };

            mkPushCmd = imageName: ''docker push "$1"/"$2"/${imageName}:"$5"'';
            pushCmd = mkPushCmd final.fractopo-app-image-stream.imageName;

          in ''
            echo "Logging in to $1 with user $4"
            docker login -p "$3" -u "$4" "$1"

            echo "Listing images"
            docker image list

            echo "Tagging new image version to $1/$2"
            ${tagCmd}

            echo "Pushing new image version to $1/$2"
            ${pushCmd}
          '';
        };
        cut-release-changelog = prev.writeShellApplication {
          name = "cut-release-changelog";
          text = ''
            ${prev.busybox}/bin/sed -n '3,/## v[[:digit:]].[[:digit:]].[[:digit:]]/p' CHANGELOG.md | head -n -2
          '';
        };
      };

  in { inherit packageOverlay localOverlay; })

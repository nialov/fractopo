{ buildPythonPackage, lib, click, pytest, geopandas, joblib, matplotlib, numpy
, pandas, rich, scikit-learn, scipy, seaborn, shapely, typer, pytest-regressions
, hypothesis, poetry-core, sphinxHook, pandoc, sphinx-autodoc-typehints
, sphinx-rtd-theme, sphinx-gallery, nbsphinx, notebook, ipython, coverage
, powerlaw, python-ternary, marimo,

}:

let

  baseFiles =
    [ ../pyproject.toml ../fractopo ../README.rst ../tests ../marimos ];
  docFiles = baseFiles ++ [ ../docs_src ../examples ];
  mkSrc = files:
    let
      fs = lib.fileset;
      sourceFiles = fs.intersection (fs.gitTracked ../.) (fs.unions files);
      src = fs.toSource {
        root = ../.;
        fileset = sourceFiles;
      };
    in src;
  self = buildPythonPackage {
    pname = "fractopo";
    version = "0.7.0";

    src = mkSrc baseFiles;

    # TODO: Conflicts when other package also includes the same file
    # nix puts both in site-packages/ directory
    # postPatch = ''
    #   substituteInPlace pyproject.toml \
    #       --replace-fail 'include = ["CHANGELOG.md"]' ""
    # '';
    format = "pyproject";

    nativeBuildInputs = [
      # Uses poetry for install
      poetry-core
    ];

    passthru = {
      # Enables building package without tests
      # nix build .#fractopo.passthru.no-check
      no-check = self.overridePythonAttrs (_: { doCheck = false; });
      # Documentation without tests
      documentation = self.overridePythonAttrs (prevAttrs: {
        src = mkSrc docFiles;
        doCheck = false;
        nativeBuildInputs = prevAttrs.nativeBuildInputs ++ [
          # Documentation dependencies
          sphinxHook
          pandoc
          sphinx-autodoc-typehints
          sphinx-rtd-theme
          sphinx-gallery
          nbsphinx
          notebook
          ipython
        ];
        sphinxRoot = "docs_src";
        outputs = [ "out" "doc" ];
      });
    };

    propagatedBuildInputs = [
      click
      geopandas
      joblib
      matplotlib
      numpy
      pandas
      powerlaw
      python-ternary
      rich
      scikit-learn
      scipy
      seaborn
      shapely
      typer
    ];

    checkInputs = [ pytest pytest-regressions hypothesis coverage marimo ];

    # TODO: Should this be precheck or does postInstall affect the docs build as well?
    postInstall = ''
      HOME="$(mktemp -d)"
      export HOME
      FRACTOPO_DISABLE_CACHE="1"
      export FRACTOPO_DISABLE_CACHE
    '';

    checkPhase = ''
      runHook preCheck
      python -m coverage run --source fractopo -m pytest --hypothesis-seed=1
      runHook postCheck
    '';

    postCheck = ''
      python -m coverage report --fail-under 70
    '';

    pythonImportsCheck = [ "fractopo" ];

    meta = with lib; {
      homepage = "https://github.com/nialov/fractopo";
      description = "Fracture Network analysis";
      license = licenses.mit;
      maintainers = [ maintainers.nialov ];
    };
  };
in self

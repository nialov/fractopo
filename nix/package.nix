{
  buildPythonPackage,
  lib,
  click,
  pytest,
  geopandas,
  joblib,
  matplotlib,
  numpy,
  pandas,
  texliveSmall,
  imagemagick,
  glibcLocales,
  pandas-stubs,
  rich,
  scikit-learn,
  scipy,
  seaborn,
  shapely,
  typer,
  pytest-regressions,
  hypothesis,
  poetry-core,
  sphinxHook,
  pandoc,
  sphinx-autodoc-typehints,
  sphinx-rtd-theme,
  sphinx-gallery,
  nbsphinx,
  sphinx-design,
  sphinx-sitemap,
  notebook,
  ipython,
  coverage,
  powerlaw,
  python-ternary,
  marimo,
  versionCheckHook,
  beartype,
  fastapi,

}:

let

  baseFiles = [
    ../pyproject.toml
    ../fractopo
    ../README.rst
    ../tests
    ../marimos
  ];
  docFiles = baseFiles ++ [
    ../docs_src
    ../examples
  ];
  mkSrc =
    files:
    let
      fs = lib.fileset;
      sourceFiles = fs.intersection (fs.gitTracked ../.) (fs.unions files);
      src = fs.toSource {
        root = ../.;
        fileset = sourceFiles;
      };
    in
    src;
  self = buildPythonPackage {
    pname = "fractopo";
    inherit ((builtins.fromTOML (builtins.readFile ../pyproject.toml)).project) version;

    src = mkSrc baseFiles;

    env = {
      # Disable disk caching during package build and testing
      FRACTOPO_DISABLE_CACHE = "1";
    };

    # TODO: Should this be precheck or does postInstall affect the docs build as well?
    postPatch = ''
      HOME="$(mktemp -d)"
      export HOME
    '';

    format = "pyproject";

    nativeBuildInputs = [
      # Uses poetry for install
      poetry-core
      pandoc
    ];

    passthru = {
      # Enables building package without tests
      # nix build .#fractopo.passthru.no-check
      no-check = self.overridePythonAttrs (_: {
        doCheck = false;
      });
      # Documentation without tests
      documentation = self.overridePythonAttrs (prevAttrs: {
        src = mkSrc docFiles;
        doCheck = false;
        dependencies = [ prevAttrs.dependencies ] ++ prevAttrs.optional-dependencies.dev;
        sphinxRoot = "docs_src";
        outputs = [
          "out"
          "doc"
        ];
        # Normal Python package build expects dist/
        preConfigure = ''
          pythonOutputDistPhase() { touch $dist; }
        '';
      });
      # PDF documentation via LaTeX/pdflatex
      documentation-pdf = self.overridePythonAttrs (prevAttrs: {
        src = mkSrc docFiles;
        doCheck = false;
        dependencies = [ prevAttrs.dependencies ] ++ prevAttrs.optional-dependencies.dev;
        nativeBuildInputs = prevAttrs.nativeBuildInputs ++ [
          texliveSmall
          imagemagick
        ];
        sphinxRoot = "docs_src";
        sphinxBuilders = "latexpdf";
        outputs = [
          "out"
          "doc"
        ];
        # Normal Python package build expects dist/
        preConfigure = ''
          pythonOutputDistPhase() { touch $dist; }
        '';
        # sphinx-hook installSphinxPhase tries .sphinx/latexpdf/latexpdf/ which does not
        # exist; the actual PDF is at .sphinx/latexpdf/latex/fractopo.pdf. Override it.
        postInstallSphinx = ''
          docdir="''${doc}/share/doc/''${name}"
          mkdir -p "$docdir/latexpdf"
          cp .sphinx/latexpdf/latex/*.pdf "$docdir/latexpdf/" || true
        '';
        # Fix locale for sphinx/pdflatex; glibcLocales provides locale-archive
        LOCALE_ARCHIVE = "${glibcLocales}/lib/locale/locale-archive";
        env = prevAttrs.env // {
          LC_ALL = "en_US.UTF-8";
        };
      });
    };

    dependencies = [
      click
      geopandas
      joblib
      matplotlib
      numpy
      pandas
      pandas-stubs
      powerlaw
      python-ternary
      rich
      scikit-learn
      scipy
      seaborn
      shapely
      typer
      beartype
    ];

    optional-dependencies = {
      dev = [
        sphinxHook
        sphinx-autodoc-typehints
        sphinx-rtd-theme
        sphinx-gallery
        nbsphinx
        sphinx-design
        sphinx-sitemap
        notebook
        ipython
        marimo
      ]
      ++ self.nativeBuildInputs;
      api = [
        fastapi
        marimo
      ];
    };

    nativeCheckInputs = [
      pytest
      pytest-regressions
      hypothesis
      coverage
      marimo
      versionCheckHook
    ];

    versionCheckProgramArg = "--version";

    checkPhase = ''
      runHook preCheck
      versionCheckHook
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
      mainProgram = "fractopo";
    };
  };
in
self

{ buildPythonPackage, fetchFromGitHub, lib, pytestCheckHook, click, pytest
, geopandas, joblib, matplotlib, numpy, pandas, pygeos, rich, scikit-learn
, scipy, seaborn, shapely, typer, pytest-regressions, hypothesis, fetchPypi
, mpmath, poetry-core, sphinxHook, pandoc, sphinx-autodoc-typehints
, sphinx-rtd-theme, sphinx-gallery, nbsphinx, notebook, ipython, coverage
, filter

# , gpgme, isPy38
}:

let
  python-ternary = buildPythonPackage rec {
    pname = "python-ternary";
    version = "1.0.8";

    src = fetchPypi {
      inherit pname version;
      sha256 =
        "41e7313db74ab2e24280797ed8073eccad4006429dfd87f6e66e7feba2aa64cd";
    };

    propagatedBuildInputs = [ matplotlib ];

    checkInputs = [ pytestCheckHook pytest ];

    pythonImportsCheck = [ "ternary" ];

    meta = with lib; {
      description = "Make ternary plots in python with matplotlib";
      homepage = "https://github.com/marcharper/python-ternary";
      license = licenses.mit;
      maintainers = with maintainers; [ nialov ];
    };
  };
  powerlaw = buildPythonPackage {
    pname = "powerlaw";
    version = "1.5";

    src = fetchFromGitHub {
      owner = "jeffalstott";
      repo = "powerlaw";
      # The version at this rev should be aligned to the one at pypi according
      # to the commit message.
      rev = "6732699d790edbe27c2790bf22c3ef7355d2b07e";
      sha256 = "sha256-x3jXk+xOQpIeEGlzYqNwuZNPpkesF0IOX8gUhhwHk5Q=";
    };

    propagatedBuildInputs = [ scipy numpy matplotlib mpmath ];

    postPatch = ''
      substituteInPlace testing/test_powerlaw.py \
          --replace "reference_data/" "testing/reference_data/"
    '';
    # --replace "reference_data/blackouts.txt" "testing/reference_data/blackouts.txt" \
    # --replace "reference_data/cities.txt" "testing/reference_data/cities.txt" \
    # --replace "reference_data/fires.txt" "testing/reference_data/fires.txt" \
    # --replace "reference_data/flares.txt" "testing/reference_data/flares.txt" \
    # --replace "reference_data/terrorism.txt" "testing/reference_data/terrorism.txt"

    checkInputs = [ pytest pytestCheckHook ];

    pytestFlagsArray = [ "testing" ];

    pythonImportsCheck = [ "powerlaw" ];

    meta = with lib; {
      description =
        "Toolbox for testing if a probability distribution fits a power law";
      homepage = "http://www.github.com/jeffalstott/powerlaw";
      license = licenses.mit;
      maintainers = with maintainers; [ nialov ];
    };
  };

in buildPythonPackage {
  pname = "fractopo";
  version = "0.5.3";

  src = filter {
    root = ./.;
    # If no include is passed, it will include all the paths.
    include = [
      # Include the "src" path relative to the root.
      "fractopo"
      "tests"
      "README.rst"
      "pyproject.toml"
      "docs_src"
      "examples"
      # Include this specific path. The path must be under the root.
      # ./package.json
      # Include all files with the .js extension
      # (filter.matchExt "js")
    ];

    # Works like include, but the reverse.
    # exclude = [ ./main.js ];
  };
  format = "pyproject";

  # Uses poetry for install
  nativeBuildInputs = [
    poetry-core
    sphinxHook
    pandoc
    sphinx-autodoc-typehints
    sphinx-rtd-theme
    sphinx-gallery
    nbsphinx
    matplotlib
    notebook
    ipython
  ];

  sphinxRoot = "docs_src";
  outputs = [ "out" "doc" ];

  propagatedBuildInputs = [
    # gpgmeOverride
    click
    geopandas
    joblib
    matplotlib
    numpy
    pandas
    powerlaw
    pygeos
    python-ternary
    rich
    scikit-learn
    scipy
    seaborn
    shapely
    typer
  ];

  checkInputs = [ pytest pytest-regressions hypothesis coverage ];

  checkPhase = ''
    runHook preCheck
    coverage run --source fractopo -m pytest
    runHook postCheck
  '';

  postCheck = ''
    coverage report --fail-under 70
  '';

  pythonImportsCheck = [ "fractopo" ];

  meta = with lib; {
    homepage = "https://github.com/nialov/fractopo";
    description = "Fracture Network analysis";
    license = licenses.mit;
    maintainers = [ maintainers.nialov ];
  };
}

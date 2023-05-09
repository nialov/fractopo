{ buildPythonPackage, fetchFromGitHub, lib, pytestCheckHook, click, pytest
, poetry2nix, geopandas, joblib, matplotlib, numpy, pandas, pygeos, rich
, scikit-learn, scipy, seaborn, shapely, typer, pytest-regressions, hypothesis
, fetchPypi, mpmath, poetry
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

    checkInputs = [ pytestCheckHook ];

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

    checkInputs = [ pytest ];

    # pytest is not actually used by the package for tests, it uses
    # unittest instead. However pytest can run all unittest cases
    # so I've just used pytest to run them with ease.
    # Tests use local files which are relative to the testing directory
    # so a cd into the testing directory was necessary for successful
    # tests.
    checkPhase = ''
      cd testing
      pytest
    '';

    pythonImportsCheck = [ "powerlaw" ];

    meta = with lib; {
      description =
        "Toolbox for testing if a probability distribution fits a power law";
      homepage = "http://www.github.com/jeffalstott/powerlaw";
      license = licenses.mit;
      maintainers = with maintainers; [ nialov ];
    };
  };
  # TODO: Fails on python38
  # gpgmeOverride = if isPy38 then
  #   null
  #   # gpgme.overrideAttrs (_: prevAttrs: {
  #   #   configureFlags = prevAttrs.configureFlags
  #   #     ++ [ ''LIBS="-L${python}/lib"'' ];
  #   #     nativeBuildInputs = prevAttrs.nativeBuildInputs ++ [
  #   #         breakpointHook
  #   #         ];
  #   # })
  # else
  #   gpgme;
  # pygeosFixed =
  #   pygeos.overrideAttrs (finalAttrs: prevAttrs: { patches = [ ]; });

in buildPythonPackage {
  pname = "fractopo";
  version = "0.5.3";
  src = let
    # Filter from src paths with given suffixes (full name can be given as suffix)
    excludeSuffixes = [
      ".flake8"
      ".nix"
      "docs_src"
      "flake.lock"
      "noxfile.py"
      "dodo.py"
      "examples"
      "environment.yml"
      "paper"
    ];
    # anyMatches returns true if any suffix matches
    anyMatches = path:
      (builtins.any (value: lib.hasSuffix value (baseNameOf path))
        excludeSuffixes);

  in builtins.filterSource (path: type:
    # Apply the anyMatches filter and reverse the result with !
    # as we want to EXCLUDE rather than INCLUDE
    !(anyMatches path)) (poetry2nix.cleanPythonSources { src = ./.; });
  # src = poetry2nix.cleanPythonSources { src = ./.; };
  format = "pyproject";

  # Uses poetry for install
  nativeBuildInputs = [ poetry ];
  # postPatch = ''
  #   substituteInPlace pyproject.toml \
  #       --replace
  # '';

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

  # Can be disabled for debugging
  # doCheck = false;
  checkInputs = [ pytestCheckHook pytest pytest-regressions hypothesis ];

  pythonImportsCheck = [ "fractopo" ];

  meta = with lib; {
    homepage = "https://github.com/nialov/fractopo";
    description = "Fracture Network analysis";
    license = licenses.mit;
    maintainers = [ maintainers.nialov ];
  };
}

{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  setuptools,
  wheel,
  deepdiff,
  future,
  gmsh,
  matplotlib,
  meshio,
  networkx,
  numba,
  numpy,
  scipy,
  seaborn,
  shapely,
  six,
  sympy,
  typing-extensions,
  isort,
  mypy,
  mypy-extensions,
  ruff,
  traitlets,
  pytest,
  pytest-cov,
  # pytest-runner,
  nix-update-script,
}:

buildPythonPackage (finalAttrs: {
  pname = "porepy";
  version = "1.13";
  pyproject = true;
  __structuredAttrs = true;

  src = fetchFromGitHub {
    owner = "pmgbergen";
    repo = "porepy";
    tag = "v${finalAttrs.version}";
    hash = "sha256-wEVpieVeYAo9JGiqFMrR/G0efz7lFjD/qoZqqvv1NYU=";
  };

  # For e.g. matplotlib
  postPatch = ''
    HOME="$(mktemp -d)"
    export HOME
  '';

  build-system = [
    setuptools
    wheel
  ];

  dependencies = [
    deepdiff
    future
    gmsh
    matplotlib
    meshio
    networkx
    numba
    numpy
    scipy
    seaborn
    shapely
    six
    sympy
    typing-extensions
  ];

  optional-dependencies = {
    development = [
      isort
      mypy
      mypy-extensions
      ruff
      traitlets
    ];
    testing = [
      pytest
      pytest-cov
      # pytest-runner
    ];
  };

  pythonRelaxDeps = [
    "gmsh"
  ];

  # TODO: Run full build with time
  # checkInputs = [ pytestCheckHook ] ++ finalAttrs.passthru.optional-dependencies.testing;

  pythonImportsCheck = [
    "porepy"
  ];

  passthru.updateScript = nix-update-script { };

  meta = {
    description = "Python Simulation Tool for Fractured and Deformable Porous Media";
    homepage = "https://github.com/pmgbergen/porepy";
    changelog = "https://github.com/pmgbergen/porepy/releases/tag/${finalAttrs.src.tag}";
    license = lib.licenses.gpl3Only;
    maintainers = with lib.maintainers; [ nialov ];
  };
})

Contributing Guidelines
=======================

Contributions to ``fractopo`` are welcome and highly appreciated.
Contributions do not have to be pull requests (i.e., code
contributions), rather it is very useful for you to report any issues
and problems you have in installing, using the software and
misunderstandings that might arise due to lacking or misleading
documentation.

If you are up for submitting code, an issue is first recommended to be
created about your plans for the submission so it can be determined if
the pull request is deemed suitable for the project and subsequently,
time will not be wasted making pull requests that are not suitable.

In particular, when submitting a pull request:

-  Install the requirements for the project using ``poetry``. ``poetry``
   uses the ``pyproject.toml`` and ``poetry.lock`` files to replicate
   the development environment.

-  Run the unit tests using ``pytest`` i.e., ``poetry run pytest``. Some
   tests are very strict (regression tests i.e. the result should match
   exactly the previous test results) and therefore no strict
   requirements are made that all tests should always pass for every
   pull request. A few failing tests can always be reviewed in the pull
   request phase!

-  Please write new tests if functionality is added or changed!

Style
-----

-  See ``pyproject.toml`` for the supported Python versions

-  You should set up `pre-commit hooks <https://pre-commit.com/>`__ to
   automatically run a number of style and syntax checks on the code.
   The ``pre-commit`` checks are the primary validation for the style of
   the code.

-  New code should be documented following the style of the code base i.e.,
   using the ``sphinx`` style.

Development dependencies
------------------------

Development dependencies for ``fractopo`` include:

-  `poetry <https://github.com/python-poetry/poetry>`__

   -  Used to handle Python package dependencies.

-  `nix <https://nixos.org/>`__

   -  ``fractopo`` is also packaged with ``nix``. ``nix`` provides
      declarative and immutable packaging which should make ``fractopo``
      last longer.

   .. code:: bash

      # To run the fractopo command-line using nix
      nix run github:nialov/fractopo#fractopo -- --help

-  `pytest <https://github.com/pytest-dev/pytest>`__

   -  ``pytest`` is a Python test runner. It is used to run defined
      tests to check that the package executes as expected. The defined
      tests in ``./tests`` contain many regression tests (done with
      ``pytest-regressions``) that make it almost impossible to add
      features to ``fractopo`` that changes (regresses) the results of
      functions and methods without notice.

-  `coverage <https://github.com/nedbat/coveragepy>`__

   -  Gathers code coverage info

-  `sphinx <https://github.com/sphinx-doc/sphinx>`__

   -  Creates documentation from files in ``./docs_src`` and
      ``examples``.

Big thanks to all maintainers of the above and other used packages!

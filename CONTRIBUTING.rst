Guidelines
==========

Contributions to ``fractopo`` are welcome and highly appreciated.
Contributions do not have to be pull requests (i.e., code
contributions), rather it is very useful for you to report any issues
and problems you have in installing, using the software and
misunderstandings that might arise due to lacking or misleading
documentation.

If you are up for submitting code, an issue is first recommended to be
created so it can be determined if the pull request is deemed suitable
for the project and subsequently, time will not be wasted making pull
requests that are not suitable.

In particular, when submitting a pull request:

-  Install the requirements for the project using ``poetry``. ``poetry``
   uses the ``pyproject.toml`` and ``poetry.lock`` files to replicate
   the development environment.

-  Run the unit tests using ``pytest`` i.e., ``poetry run pytest``.
   Some tests are very strict (regression tests, result should match
   exactly the previous test results) and therefore no strict
   requirements are made that all tests should always pass
   for every pull request. A few failing tests can always be reviewed
   in the pull request phase!

-  Please write new tests if functionality is added or changed!

Style
-----

-  See ``pyproject.toml`` for the supported ``Python`` versions

-  You should set up `pre-commit hooks <https://pre-commit.com/>`__ to
   automatically run a number of style and syntax checks on the code.
   The ``pre-commit`` checks are the primary validation for the style of
   the code.

-  New code should be documented following the style of the code base i.e.,
   using the ``sphinx`` style.

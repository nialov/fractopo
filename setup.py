"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
Modified by Madoshakalaka@Github (dependency links added)
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name="fractopo",  # Required
    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="0.0.1",  # Required
    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description="Brittle Geology Analysis Toolkit",  # Optional
    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    #
    # Often, this is the same as your README, so you can just read it in from
    # that file directly (as we have already done above)
    #
    # This field corresponds to the "Description" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    long_description=long_description,  # Optional
    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    #
    # Optional if long_description is written in reStructuredText (rst) but
    # required for plain-text or Markdown; if unspecified, "applications should
    # attempt to render [the long_description] as text/x-rst; charset=UTF-8 and
    # fall back to text/plain if it is not valid rst" (see link below)
    #
    # This field corresponds to the "Description-Content-Type" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    # This should be a valid link to your project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url="https://github.com/nialov/fractopo",  # Optional
    # This should be your name or the name of the organization which owns the
    # project.
    author="Nikolas Ovaskainen",  # Optional
    # This should be a valid email address corresponding to the author listed
    # above.
    author_email="nikolasovaskainen@gmail.com",  # Optional
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Brittle Geology Data Analysis :: Data Analysis",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # These classifiers are *not* checked by 'pip install'. See instead
        # 'python_requires' below.
        "Programming Language :: Python :: 3.8",
    ],
    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a string of words separated by whitespace, not a list.
    keywords="sample setuptools development",  # Optional
    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_packages(exclude=["contrib", "docs", "tests"]),  # Required
    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. If you
    # do not support Python 2, you can simplify this to '>=3.5' or similar, see
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires=">=3.8",
    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "attrs==20.3.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "certifi==2020.6.20",
        "click==7.1.2",
        "click-plugins==1.1.1",
        "cligj==0.7.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3' and python_version < '4'",
        "cycler==0.10.0",
        "et-xmlfile==1.0.1",
        "fiona==1.8.17",
        "geopandas==0.8.1",
        "geotrans==0.0.4",
        "jdcal==1.4.1",
        "joblib==1.0.0; python_version >= '3.6'",
        "kiwisolver==1.3.1; python_version >= '3.6'",
        "matplotlib==3.3.3",
        "mpmath==1.1.0",
        "munch==2.5.0",
        "numpy==1.19.4",
        "openpyxl==3.0.5",
        "pandas==1.1.4",
        "pillow==8.0.1; python_version >= '3.6'",
        "powerlaw==1.4.6",
        "pygeos==0.8",
        "pyparsing==2.4.7; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "pyproj==3.0.0.post1; python_version >= '3.6'",
        "python-dateutil==2.8.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "python-ternary==1.0.7",
        "pytz==2020.4",
        "scikit-learn==0.23.2; python_version >= '3.6'",
        "scipy==1.5.4",
        "seaborn==0.11.0",
        "shapely==1.7.1",
        "six==1.15.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "sklearn==0.0",
        "threadpoolctl==2.1.0; python_version >= '3.5'",
        "xlrd==2.0.1",
    ],  # Optional
    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    extras_require={
        "dev": [
            "alabaster==0.7.12",
            "appdirs==1.4.4",
            "argon2-cffi==20.1.0",
            "async-generator==1.10; python_version >= '3.5'",
            "attrs==20.3.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "babel==2.9.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "backcall==0.2.0",
            "bentley-ottmann==0.9.0; python_version >= '3.5'",
            "black==20.8b1",
            "bleach==3.2.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
            "certifi==2020.6.20",
            "cffi==1.14.4",
            "chardet==4.0.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
            "click==7.1.2",
            "commonmark==0.9.1",
            "coverage==5.3",
            "cycler==0.10.0",
            "decision==0.2.0; python_version >= '3.5'",
            "decorator==4.4.2",
            "defusedxml==0.6.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
            "dendroid==1.1.0; python_version >= '3.5'",
            "descartes==1.1.0",
            "distlib==0.3.1",
            "docutils==0.16; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
            "entrypoints==0.3; python_version >= '2.7'",
            "filelock==3.0.12",
            "hypothesis==5.43.3",
            "hypothesis-geometry==0.17.1",
            "idna==2.10; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "imagesize==1.2.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "iniconfig==1.1.1",
            "ipykernel==5.4.2; python_version >= '3.5'",
            "ipython==7.19.0; python_version >= '3.7'",
            "ipython-genutils==0.2.0",
            "jedi==0.17.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
            "jinja2==2.11.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
            "json5==0.9.5",
            "jsonschema==3.2.0",
            "jupyter-client==6.1.7; python_version >= '3.5'",
            "jupyter-core==4.7.0; python_version >= '3.6'",
            "jupyterlab==2.2.9",
            "jupyterlab-pygments==0.1.2",
            "jupyterlab-server==1.2.0; python_version >= '3.5'",
            "kiwisolver==1.3.1; python_version >= '3.6'",
            "locus==0.7.1; python_version >= '3.5'",
            "markupsafe==1.1.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "matplotlib==3.3.3",
            "mistune==0.8.4",
            "mypy==0.790",
            "mypy-extensions==0.4.3",
            "nbclient==0.5.1; python_version >= '3.6'",
            "nbconvert==6.0.7; python_version >= '3.6'",
            "nbformat==5.0.8; python_version >= '3.5'",
            "nbsphinx==0.8.0",
            "nest-asyncio==1.4.3; python_version >= '3.5'",
            "notebook==6.1.5; python_version >= '3.5'",
            "numpy==1.19.4",
            "packaging==20.8; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "pandocfilters==1.4.3",
            "parso==0.7.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "pathspec==0.8.1",
            "pbr==5.5.1; python_version >= '2.6'",
            "pexpect==4.8.0; sys_platform != 'win32'",
            "pickleshare==0.7.5",
            "pillow==8.0.1; python_version >= '3.6'",
            "pipenv==2020.11.15; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "pipenv-to-requirements==0.9.0",
            "pluggy==0.13.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "prioq==0.3.0; python_version >= '3.5'",
            "prometheus-client==0.9.0",
            "prompt-toolkit==3.0.8; python_full_version >= '3.6.1'",
            "ptyprocess==0.6.0; os_name != 'nt'",
            "py==1.10.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "pycparser==2.20; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "pygments==2.7.3; python_version >= '3.5'",
            "pyparsing==2.4.7; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "pyrsistent==0.17.3; python_version >= '3.5'",
            "pytest==6.2.1",
            "pytest-datadir==1.3.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "pytest-regressions==2.1.1",
            "python-dateutil==2.8.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "pytz==2020.4",
            "pyyaml==5.3.1",
            "pyzmq==20.0.0; python_version >= '3.5'",
            "recommonmark==0.7.1",
            "regex==2020.11.13",
            "reprit==0.3.1; python_full_version >= '3.5.3'",
            "requests==2.25.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
            "robust==0.2.7; python_version >= '3.5'",
            "send2trash==1.5.0",
            "six==1.15.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "snowballstemmer==2.0.0",
            "sortedcontainers==2.3.0",
            "sphinx==3.3.1; python_version >= '3.5'",
            "sphinx-rtd-theme==0.5.0",
            "sphinxcontrib-applehelp==1.0.2; python_version >= '3.5'",
            "sphinxcontrib-devhelp==1.0.2; python_version >= '3.5'",
            "sphinxcontrib-htmlhelp==1.0.3; python_version >= '3.5'",
            "sphinxcontrib-jsmath==1.0.1; python_version >= '3.5'",
            "sphinxcontrib-qthelp==1.0.3; python_version >= '3.5'",
            "sphinxcontrib-serializinghtml==1.1.4; python_version >= '3.5'",
            "terminado==0.9.1; python_version >= '3.6'",
            "testpath==0.4.4",
            "toml==0.10.2; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "tornado==6.1; python_version >= '3.5'",
            "tox==3.20.1",
            "traitlets==5.0.5; python_version >= '3.7'",
            "typed-ast==1.4.1",
            "typing-extensions==3.7.4.3",
            "urllib3==1.26.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4' and python_version < '4'",
            "virtualenv==20.2.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "virtualenv-clone==0.5.4; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "wcwidth==0.2.5",
            "webencodings==0.5.1",
        ]
    },  # Optional
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    #
    # Sometimes you’ll want to use packages that are properly arranged with
    # setuptools, but are not published to PyPI. In those cases, you can specify
    # a list of one or more dependency_links URLs where the package can
    # be downloaded, along with some additional hints, and setuptools
    # will find and install the package correctly.
    # see https://python-packaging.readthedocs.io/en/latest/dependencies.html#packages-not-on-pypi
    #
    dependency_links=[],
    # If using Python 2.6 or earlier, then these have to be included in
    # MANIFEST.in as well.
    # package_data={"sample": ["package_data.dat"]},  # Optional
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[("my_data", ["data/data_file"])],  # Optional
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    entry_points="""
        [console_scripts]
        tracevalidate=fractopo.cli:tracevalidate
    """,
    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        "Bug Reports": "https://github.com/nialov/fractopo/issues",
        "Source": "https://github.com/nialov/fractopo/",
    },
)

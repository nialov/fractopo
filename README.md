# Fractopo-2D

Fractopo-2D is a Python module that contains tools for validating and analysing
lineament and fracture trace maps (fracture networks).

![Overview of fractopo-2D](docs_src/imgs/fractopo_2d_diagram.png)

## Development status

* In development.

## Full documentation

* Documentation hosted on GitHub pages:
  * [Documentation](https://nialov.github.io/fractopo/index.html)

## Installation

### Pipenv

~~~bash
git clone https://github.com/nialov/fractopo --depth 1
cd fractopo
pipenv sync --dev
~~~

### Pip

The module is not on pypi currently. But pip can install from github.

~~~bash
pip install git+https://github.com/nialov/fractopo#egg=fractopo
~~~

## Usage

Usage guide is WIP.

### Trace validation

Trace validation is accessible as a console script, `tracevalidate`

Trace validation always requires the target area that delineates trace data.

~~~bash
# Get script help

tracevalidate --help

# Basic usage:
# --fix is recommended, probably won't work without it.
# --output can be omitted. By default the same spatial filetype
# as the input is used and the output is saved as e.g.
# /path/to/trace_data_validated.shp

tracevalidate /path/to/trace_data.shp /path/to/target_area.shp --fix --output /path/to/output_data.shp
~~~

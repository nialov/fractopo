fractopo
========

`fractopo` is a Python module that contains tools for validating and
analysing lineament and fracture trace maps (fracture networks).

.. figure:: docs_src/imgs/fractopo_2d_diagram.png
   :alt: Overview of fractopo-2D

   Overview of fractopo

Development status
------------------

-  In constant development, will have breaking changes.

-  Critical issues:

   -  ``snap_traces`` in branch and node determination is not completely
      stable. Some edge cases cause artifacts which only sometimes are
      recognized as error branches. (Mostly solved as of 1.3.2021).

      -  Reinforces that some amount of responsibility is always in the
         hands of the digitizer.

      -  Issue mostly avoided with a ``snap_threshold`` of 0.001

Full documentation
------------------

-  Documentation hosted on Read the Docs:

   -  `Documentation <https://fractopo.readthedocs.io/en/latest/index.html>`__

Installation
------------

Omit ``--dev`` or ``[dev]`` for regular installation. Keep if you want
to test/develop or otherwise install all development python
dependencies.

Pipenv
~~~~~~

.. code:: bash

   git clone https://github.com/nialov/fractopo --depth 1
   cd fractopo
   # Omit --dev from end if you do not want installation for development
   pipenv sync --dev

Pip
~~~

The module is not on pypi currently. But pip can install from github.

.. code:: bash

   # Non-development installation
   pip install git+https://github.com/nialov/fractopo#egg=fractopo

Or locally

.. code:: bash

   git clone https://github.com/nialov/fractopo --depth 1
   cd fractopo
   # Omit [dev] from end if you do not want installation for development
   pip install --editable .[dev]

Usage
-----

See `Notebooks with examples <https://tinyurl.com/yb4tj47e>`__ for more
advanced usage guidance and examples.

Input data
~~~~~~~~~~

Reading and writing spatial filetypes is done in `geopandas` and you should see
`geopandas` documentation for more advanced read-write use cases:

-  https://geopandas.org/

Simple example with trace and area data in GeoPackages:

.. code:: python

   import geopandas as gpd

   # Trace data is in a file `traces.gpkg` in current working directory
   # Area data is in a file `areas.gpkg` in current working directory
   trace_data = gpd.read_file("traces.gpkg")
   area_data = gpd.read_file("areas.gpkg")

Geometric and topological trace network analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trace and target area data (``GeoDataFrame``\ s) are passed into a
``Network`` object which has properties and functions for returning and
visualizing different parameters and attributes of trace data.

.. code:: python

   from fractopo.analysis.network import Network
   network = Network(
       trace_data, area_data, name="mynetwork", determine_branches_nodes=True,
   )

   # Properties are easily accessible
   # e.g.
   network.branch_counts
   network.node_counts

   # Plotting is done by plot_ -prefixed methods
   network.plot_trace_lengths()

Trace validation
~~~~~~~~~~~~~~~~

Trace and target area data can be validated for further analysis with a
``Validation`` object.

.. code:: python

   from fractopo.tval.trace_validation import Validation
   validation = Validation(
       trace_data, area_data, name="mytraces", allow_fix=True,
   )

   # Validation is done explicitly with `run_validation` method
   validated_trace_data = validation.run_validation()

Trace validation is also accessible as a command-line script,
``tracevalidate`` which is more straightforward to use than through
Python calls.

``tracevalidate`` always requires the target area that delineates trace
data.

.. code:: bash

   # Get full up-to-date script help

   tracevalidate --help

   # Basic usage:
   # --fix is recommended due to automatic fixing being very minor in effect
   # currently
   # --output can be omitted. By default the same spatial filetype
   # as the input is used and the output is saved as e.g.
   # /path/to/validated/trace_data_validated.shp
   # i.e. a new folder is created (or used) for validated data
   # --summary can be given to print out summary data of validation
   # i.e. error types and error counts

   tracevalidate /path/to/trace_data.shp /path/to/target_area.shp --fix --output /path/to/output_data.shp

   # Or with automatic saving to validated/ directory

   tracevalidate /path/to/trace_data.shp /path/to/target_area.shp --fix --summary

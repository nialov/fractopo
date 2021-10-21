fractopo-readme
===============

|Documentation Status| |PyPI Status| |CI Test| |Coverage| |Binder| |Zenodo|

``fractopo`` is a Python module that contains tools for validating and
analysing lineament and fracture trace maps (fracture networks).

.. figure:: https://git.io/JBRuK
   :alt: Overview of fractopo

   Overview of fractopo

-  Full Documentation is hosted on Read the Docs:

   -  `Documentation <https://fractopo.readthedocs.io/en/latest/index.html#full-documentation>`__

Installation
------------

``pip`` and ``poetry`` installation only supported for ``linux`` and
``MacOS`` based operating systems. For ``Windows`` install using
``(ana)conda``.

For ``pip`` and ``poetry``: Omit --dev or [dev] for regular
installation. Keep if you want to test/develop or otherwise install all
development python dependencies.

Conda
~~~~~

-  Only supported installation method for ``Windows``!

.. code:: bash

   # Create new environment for fractopo (recommended)
   conda env create fractopo-env
   conda activate fractopo-env
   # Available on conda-forge channel
   conda install -c conda-forge fractopo

Pip
~~~

The module is on `PyPI <https://www.pypi.org>`__.

.. code:: bash

   # Non-development installation
   pip install fractopo

Or locally for development:

.. code:: bash

   git clone https://github.com/nialov/fractopo
   cd fractopo
   # Omit [dev] from end if you do not want installation for development
   pip install --editable .[dev]

poetry
~~~~~~

For usage:

.. code:: bash

   poetry add fractopo

For development:

.. code:: bash

   git clone https://github.com/nialov/fractopo --depth 1
   cd fractopo
   poetry install

Input data
~~~~~~~~~~

Reading and writing spatial filetypes is done in ``geopandas`` and you
should see ``geopandas`` documentation for more advanced read-write use
cases:

-  https://geopandas.org/

Simple example with trace and area data in GeoPackages:

.. code:: python

   import geopandas as gpd

   # Trace data is in a file `traces.gpkg` in current working directory
   # Area data is in a file `areas.gpkg` in current working directory
   trace_data = gpd.read_file("traces.gpkg")
   area_data = gpd.read_file("areas.gpkg")

Trace validation
~~~~~~~~~~~~~~~~

Trace and target area data can be validated for further analysis with a
``Validation`` object.

.. code:: python

   from fractopo import Validation

   validation = Validation(
       trace_data,
       area_data,
       name="mytraces",
       allow_fix=True,
   )

   # Validation is done explicitly with `run_validation` method
   validated_trace_data = validation.run_validation()

Trace validation is also accessible as a command-line script,
``fractopo tracevalidate`` which is more straightforward to use than through
Python calls. Note that all subcommands of ``fractopo`` are available by
appending them after ``fractopo``.

``tracevalidate`` always requires the target area that delineates trace
data.

.. code:: bash

   # Get full up-to-date script help

   fractopo tracevalidate --help

   # Basic usage:
   # --allow-fix is recommended due to automatic fixing being very minor in effect
   # currently (default True)
   # --summary can be given to print out summary data of validation
   # i.e. error types and error counts (default True)
   # --output can be omitted. By default the same spatial filetype
   # as the input is used and the output is saved as e.g.
   # /path/to/validated/trace_data_validated.shp
   # i.e. a new folder is created (or used) for validated data

   fractopo tracevalidate /path/to/trace_data.shp /path/to/target_area.shp --fix --output /path/to/output_data.shp

   # Or with automatic saving to validated/ directory

   fractopo tracevalidate /path/to/trace_data.shp /path/to/target_area.shp --fix --summary

Geometric and topological trace network analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trace and target area data (``GeoDataFrames``) are passed into a
``Network`` object which has properties and functions for returning and
visualizing different parameters and attributes of trace data.

.. code:: python

   from fractopo import Network

   # Initialize Network object and determine the topological branches and nodes
   network = Network(
       trace_data,
       area_data,
       # Give the Network a name!
       name="mynetwork",
       # Specify whether to determine topological branches and nodes
       # (Required for almost all analysis)
       determine_branches_nodes=True,
       # Specify the snapping distance threshold to define when traces are
       # snapped to each other
       snap_threshold=0.001,
       # If the target area used in digitization is a circle, the knowledge can
       # be used in some analysis
       circular_target_area=True,
       # Analysis on traces can be done for the full inputted dataset or the
       # traces can be cropped to the target area before analysis (cropping
       # recommended)
       truncate_traces=True,
   )

   # Properties are easily accessible
   # e.g.
   network.branch_counts
   network.node_counts

   # Plotting is done by plot_ -prefixed methods
   network.plot_trace_lengths()

Network analysis is also available as a command-line script but I recommend
using a Python interface (e.g. ``jupyter lab``, ``ipython``) when analysing
``Networks`` to have access to all available analysis and plotting methods. The
command-line entrypoint is opinionated in what outputs it produces. Brief
example of command-line entrypoint:

.. code:: bash

   fractopo network traces.gpkg area.gpkg --name mynetwork\
      --circular-target-area --truncate-traces

   # Use --help to see all up-to-date arguments and help
   fractopo network --help

Development status
------------------

-  Breaking changes are possible and expected.
-  Critical issues:

   -  Trace validation should be refactored at some point.

      -  Though keeping in mind that the current implementation works
         well.

   -  ``snap_traces`` in branch and node determination is not perfect.
      Some edge cases cause artifacts which only sometimes are
      recognized as error branches. However these cases are very rare.

      -  Reinforces that some amount of responsibility is always in the
         hands of the digitizer.
      -  Issue mostly avoided when using a ``snap_threshold`` of 0.001


.. |Documentation Status| image:: https://readthedocs.org/projects/fractopo/badge/?version=latest
   :target: https://fractopo.readthedocs.io/en/latest/?badge=latest
.. |PyPI Status| image:: https://img.shields.io/pypi/v/fractopo.svg
   :target: https://pypi.python.org/pypi/fractopo
.. |CI Test| image:: https://github.com/nialov/fractopo/workflows/test-and-publish/badge.svg
   :target: https://github.com/nialov/fractopo/actions/workflows/test-and-publish.yaml?query=branch%3Amaster
.. |Coverage| image:: https://raw.githubusercontent.com/nialov/fractopo/master/docs_src/imgs/coverage.svg
   :target: https://github.com/nialov/fractopo/blob/master/docs_src/imgs/coverage.svg
.. |Binder| image:: http://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/nialov/fractopo/HEAD?filepath=docs_src%2Fnotebooks%2Ffractopo_network_1.ipynb
.. |Zenodo| image:: https://zenodo.org/badge/297451015.svg
   :target: https://zenodo.org/badge/latestdoi/297451015

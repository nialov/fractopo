.. Readme is included with cogapp
.. It fixes relative links to image files in the repository
.. [[[cog
   from pathlib import Path

   import cog

   cog.outl("")
   readme = Path("README.rst")
   for line in readme.read_text().splitlines():
       # Make sure this functionality fits the project documentation structure
       if "figure::" in line:
           line = line.replace("docs_src/", "")
       line = line.replace("</CONTRIBUTING.rst>" , "</CONTRIBUTING.html>")
       cog.outl(line)
   ]]]

fractopo
========

|PyPI Status| |CI Test| |Conda Test| |Binder| |Zenodo| |JOSS| |Conda Version|

``fractopo`` is a Python library/application that contains tools for
validating and analysing lineament and fracture trace maps (fracture
networks). It is targeted at structural geologists working on the
characterization of bedrock fractures from outcrops and through remote
sensing. ``fractopo`` is available as a Python library and through a
command-line interface. As a Python library, the use of ``fractopo``
requires prior (Python) programming knowledge. However, if used through
the command-line, using ``fractopo`` only requires general knowledge of
command-line interfaces in your operating system of choice.

-  `Full Documentation is hosted on GitHub
   <https://nialov.github.io/fractopo/index.html#full-documentation>`__

.. figure:: https://git.io/JBRuK
   :alt: Overview of fractopo

   Overview of fractopo

.. figure:: /imgs/fractopo-visualizations.png
   :alt: Data visualization

   Visualisation of ``fractopo`` data. ``fractopo`` analyses the trace
   data that can e.g. be digitized from drone orthophotographs
   (=fractures) or from digital elevation models (=lineaments). The
   displayed branches and nodes are extracted with ``fractopo``.

Installation
------------

``pip`` and ``poetry`` installation only supported for ``linux`` -based
operating systems. For Windows and MacOS install using
`(ana)conda <#conda>`__. A container exists for exposing a simplified
web interface using `marimo <https://github.com/marimo-team/marimo>`__.

conda
~~~~~

-  Only (supported) installation method for ``Windows`` and ``MacOS``!

.. code:: bash

   # Create new environment for fractopo (recommended but optional)
   conda env create -n fractopo-env
   conda activate fractopo-env
   # Available on conda-forge channel
   conda install -c conda-forge fractopo

pip
~~~

The module is on `PyPI <https://www.pypi.org>`__.

.. code:: bash

   # Non-development installation
   pip install fractopo

poetry
~~~~~~

For usage:

.. code:: bash

   poetry add fractopo

For development, only ``poetry`` installation of ``fractopo`` is
supported:

.. code:: bash

   git clone https://github.com/nialov/fractopo
   cd fractopo
   poetry install --all-extras

container
~~~~~~~~~

Two `marimo <https://github.com/marimo-team/marimo>`__ apps exist in
``./marimos``, ``validation.py`` and ``network.py``. These are both
included in the ``ghcr.io/nialov/fractopo-app:latest`` image. By
default, when the image is run, the network analysis app is run. To run
the validation app, add an environment variable ``RUN_MODE`` to the
container with a value of ``validation``.

The app host and port can be chosen with ``HOST`` (default is
``0.0.0.0``) and ``PORT`` (default is ``2718``) environment variables.
With ``docker`` you can, of course, expose in whichever port you choose.

Example ``docker`` invocations are below.

To run network analysis:

.. code:: bash

   # RUN_MODE=network specified explicitly even if it is the default
   docker run --rm --interactive --tty --publish 2718:2718 --env RUN_MODE=network ghcr.io/nialov/fractopo-app:latest

To run validation:

.. code:: bash

   docker run --rm --interactive --tty --publish 2718:2718 --env RUN_MODE=validation ghcr.io/nialov/fractopo-app:latest

The app should be available at http://localhost:2718

Usage
-----

``fractopo`` has two main use cases:

1. Validation of lineament & fracture trace data
2. Analysis of lineament & fracture trace data

Validation is done to make sure the data is valid for the analysis and
is crucial as analysis cannot take into account different kinds of
geometric and topological inconsistencies between the traces.
Capabilities and associated guides are inexhaustibly listed in the
table below.

========================================================  ======================
Functionality                                             Tutorial/Guide/Example
========================================================  ======================
Validation of trace data                                  `Validation 1`_; `Validation 2`_
Visualize trace map data                                  `Visualizing`_
Topological branches and nodes                            `Network`_; `Topological`_
Trace and branch length distributions                     `Length-distributions`_
Orientation rose plots                                    `Orientation 1`_; `Orientation 2`_
Plot topological ternary node and branch proportions      `Proportions`_
Cross-cutting and abutting relationships                  `Relationships 1`_; `Relationships 2`_;
Geometric and topological fracture network parameters     `Parameters`_
Contour grids of fracture network parameters              `Contour-grids`_
Multi-scale length distributions                          `Multi-scale`_
========================================================  ======================

.. _Validation 1:
   https://nialov.github.io/fractopo/notebooks/fractopo_validation_1.html
.. _Validation 2:
   https://nialov.github.io/fractopo/notebooks/fractopo_validation_2.html
.. _Visualizing:
   https://nialov.github.io/fractopo/notebooks/fractopo_network_1.html#Visualizing-trace-map-data
.. _Network:
   https://nialov.github.io/fractopo/notebooks/fractopo_network_1.html#Network
.. _Topological:
   https://nialov.github.io/fractopo/auto_examples/plot_branches_and_nodes.html#sphx-glr-auto-examples-plot-branches-and-nodes-py
.. _Length-distributions:
   https://nialov.github.io/fractopo/notebooks/fractopo_network_1.html#Length-distributions
.. _Orientation 1:
   https://nialov.github.io/fractopo/notebooks/fractopo_network_1.html#Rose-plots
.. _Orientation 2:
   https://nialov.github.io/fractopo/auto_examples/plot_rose_plot.html#sphx-glr-auto-examples-plot-rose-plot-py
.. _Proportions:
   https://nialov.github.io/fractopo/notebooks/fractopo_network_1.html#Node-and-branch-proportions
.. _Relationships 1:
   https://nialov.github.io/fractopo/notebooks/fractopo_network_1.html#Crosscutting-and-abutting-relationships
.. _Relationships 2:
   https://nialov.github.io/fractopo/auto_examples/plot_azimuth_set_relationships.html#sphx-glr-auto-examples-plot-azimuth-set-relationships-py
.. _Parameters:
   https://nialov.github.io/fractopo/notebooks/fractopo_network_1.html#Numerical-Fracture-Network-Characterization-Parameters
.. _Contour-grids:
   https://nialov.github.io/fractopo/notebooks/fractopo_network_1.html#Contour-Grids
.. _Multi-scale:
   https://nialov.github.io/fractopo/auto_examples/plot_multi_scale_networks.html#sphx-glr-auto-examples-plot-multi-scale-networks-py

For a short tutorial on use of ``fractopo`` continue reading:

Input data
~~~~~~~~~~

Reading and writing spatial filetypes is done in ``geopandas`` and you
should see ``geopandas`` documentation for more advanced read-write use
cases:

-  https://geopandas.org/

Simple example with trace and area data in ``GeoPackages``:

.. code:: python

   import geopandas as gpd

   # Trace data is in a file `traces.gpkg` in current working directory
   # Area data is in a file `areas.gpkg` in current working directory
   trace_data = gpd.read_file("traces.gpkg")
   area_data = gpd.read_file("areas.gpkg")

Trace data should consists of polyline geometries, i.e., of
``LineString`` type. Trace data in ``MultiLineString`` format area not
supported. Area data should consists of polygon geometries, i.e., of
either ``Polygon`` or ``MultiPolygon`` type.

Trace validation
~~~~~~~~~~~~~~~~

Trace data must be validated using ``fractopo`` validation functionality
before analysis. The topological analysis of lineament & fracture traces
implemented in ``fractopo`` will not tolerate uncertainty related to the
topological abutting and snapping relationships between traces. See `the
documentation <https://nialov.github.io/fractopo/validation/errors.html>`__
for further info on validation error types. Trace validation is
recommended before all analysis using ``Network``. Trace and target area
data can be validated for further analysis with a ``Validation`` object:

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

Trace validation is also accessible through the ``fractopo``
command-line interface, ``fractopo tracevalidate`` which is more
straightforward to use than through Python calls. Note that all
subcommands of ``fractopo`` are available by appending them after
``fractopo``.

``tracevalidate`` always requires the target area that delineates trace
data.

.. code:: bash

   # Get full up-to-date command-line interface help
   fractopo tracevalidate --help

   # Basic usage example:
   fractopo tracevalidate /path/to/trace_data.shp /path/to/target_area.shp\
      --output /path/to/validated_trace_data.shp

   # Or with automatic saving to validated/ directory
   fractopo tracevalidate /path/to/trace_data.shp /path/to/target_area.shp\
      --summary

Geometric and topological trace network analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fractopo`` can be used to extract lineament & fracture size,
abundance and topological parameters from two-dimensional lineament and
fracture trace, branch and node data.

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
       # snapped to each other. The unit is the same as the one in the
       # coordinate system the trace and area data are in.
       # In default values, fractopo assumes a metric unit and using metric units
       # is heavily recommended.
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
   # e.g.,
   network.branch_counts
   network.node_counts

   # Plotting is done by plot_ -prefixed methods
   network.plot_trace_lengths()

Network analysis is also available through the ``fractopo`` command-line
interface but using the Python interface (e.g. ``jupyter lab``,
``ipython``) is recommended when analysing ``Networks`` to have access
to all available analysis and plotting methods. The command-line
entrypoint is **opinionated** in what outputs it produces. Brief example
of command-line entrypoint:

.. code:: bash

   fractopo network /path/to/trace_data.shp /path/to/area_data.shp\
      --name mynetwork

   # Use --help to see all up-to-date arguments and help
   fractopo network --help

.. figure:: /imgs/fractopo_workflow_visualisation.jpg
   :alt: Data analysis workflow visualisation for fracture trace data.

   Data analysis workflow visualisation for fracture trace data
   (``KB11``). A. Target area for trace digitisation. B. Digitized
   traces and target area. C. Orthomosaic used as the base raster from
   which the traces are digitized from. D. Equal-area length-weighted
   rose plot of the fracture trace azimuths. E. Length distribution
   analysis of the trace lengths. F. Determined branches and nodes
   through topological analysis. G. Cross-cut and abutting relationships
   between chosen azimuth sets. H. Ternary plot of node (X, Y and I)
   proportions. I. Ternary plot of branch (C-C, C-I, I-I) proportions.

Citing
------

To cite this software:

.. code:: text

   Ovaskainen, N., (2023). fractopo: A Python package for fracture
   network analysis. Journal of Open Source Software, 8(85), 5300,
   https://doi.org/10.21105/joss.05300

-  To cite a specific version of ``fractopo`` you can use a ``zenodo``
   provided ``DOI``. E.g. https://doi.org/10.5281/zenodo.5957206 for version
   ``v0.2.6``. See the ``zenodo`` page of ``fractopo`` for the ``DOI`` of each
   version: https://doi.org/10.5281/zenodo.5517485

Support
-------

For issues of any kind: please create a GitHub issue here!
Alternatively, you can contact the main developer by email at
nikolasovaskainen@gmail.com.

References
----------

For the scientific background, prior works, definition of traces, branches and
nodes along with the explanation of the plots and the plotted parameters, you
are referred to multiple sources:

-  `Sanderson and Nixon,
   2015 <https://doi.org/10.1016/j.jsg.2015.01.005>`__

   -  Trace and branch size, abundance and topological parameter
      definitions.

-  `Ovaskainen et al, 2022 <https://doi.org/10.1016/j.jsg.2022.104528>`__

   -  Application of ``fractopo`` for subsampling analysis of fracture networks.

-  `Nyberg et al., 2018 <https://doi.org/10.1130/GES01595.1>`__

   -  A similar package to ``fractopo`` with a ``QGIS`` GUI.
   -  `NetworkGT GitHub <https://github.com/BjornNyberg/NetworkGT>`__

-  `Sanderson and Peacock,
   2020 <https://www.sciencedirect.com/science/article/abs/pii/S001282521930594X>`__

   -  Discussion around rose plots and justification for using
      length-weighted equal-area rose plots.

-  `Alstott et al.
   2014 <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0085777>`__

   -  Length distribution modelling using the Python 3 ``powerlaw``
      package which ``fractopo`` uses
   -  `powerlaw GitHub <https://github.com/jeffalstott/powerlaw>`__

-  `Bonnet et al.,
   2001 <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/1999RG000074>`__

   -  Length distribution modelling review.

-  `My Master’s Thesis, Ovaskainen,
   2020 <http://urn.fi/URN:NBN:fi-fe202003259211>`__

   -  Plots used in my Thesis were done with an older version of the
      same code used for this plugin.

Development
-----------

-  The package interfaces are nearing stability and breaking changes in
   code should for the most part be included in the ``CHANGELOG.md``
   after 25.4.2023. However, this is not guaranteed until the version
   reaches v1.0.0. The interfaces of ``Network`` and ``Validation`` can
   be expected to be the most stable.

-  For general contributing guidelines, see `CONTRIBUTING.rst </CONTRIBUTING.html>`__

License
~~~~~~~

Copyright © 2020-2025, Nikolas Ovaskainen.

-----

.. |PyPI Status| image:: https://img.shields.io/pypi/v/fractopo.svg
   :target: https://pypi.python.org/pypi/fractopo
.. |Conda Version| image:: https://img.shields.io/conda/vn/conda-forge/fractopo.svg
   :target: https://anaconda.org/conda-forge/fractopo
.. .. |Documentation Status| image:: https://github.com/nialov/fractopo/actions/workflows/main.yaml/badge.svg
..    :target: https://nialov.github.io/fractopo/
.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.05300/status.svg
   :target: https://doi.org/10.21105/joss.05300
.. |CI Test| image:: https://github.com/nialov/fractopo/actions/workflows/main.yaml/badge.svg
   :target: https://github.com/nialov/fractopo/actions/workflows/main.yaml?query=branch%3Amaster
.. |Conda Test| image:: https://github.com/nialov/fractopo/actions/workflows/conda.yaml/badge.svg
   :target: https://github.com/nialov/fractopo/actions/workflows/conda.yaml?query=branch%3Amaster
.. |Binder| image:: http://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/nialov/fractopo/HEAD?filepath=docs_src%2Fnotebooks%2Ffractopo_network_1.ipynb
.. |Zenodo| image:: https://zenodo.org/badge/297451015.svg
   :target: https://zenodo.org/badge/latestdoi/297451015
.. [[[end]]] (checksum: 1cc99229567da747869edfb07443524d)

.. toctree::
   :hidden:
   :caption: Links

   Documentation (You Are Here) <self>
   Homepage (GitHub) <https://github.com/nialov/fractopo>
   Bug & Issue Reporting (GitHub) <https://github.com/nialov/fractopo/issues>

.. toctree::
   :maxdepth: 1
   :caption: Notebooks
   :hidden:

   notebooks/fractopo_network_1
   notebooks/fractopo_validation_1
   notebooks/fractopo_validation_2

.. toctree::
   :maxdepth: 1
   :caption: Gallery
   :hidden:

   auto_examples/index

.. toctree::
   :maxdepth: 1
   :caption: Advanced
   :hidden:

   validation/basics
   validation/errors
   misc

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   CONTRIBUTING

.. toctree::
   :maxdepth: 1
   :caption: Module documentation

   apidoc/fractopo

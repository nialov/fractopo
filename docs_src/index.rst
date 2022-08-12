.. include:: ../README.rst

.. toctree::
   :hidden:
   :caption: Links

   Documentation (You Are Here) <self>
   Homepage (GitHub) <https://github.com/nialov/fractopo>
   Bug & Issue Reporting (GitHub) <https://github.com/nialov/fractopo/issues>

Additional Documentation
========================

.. figure:: ./imgs/fractopo-visualizations.png
   :alt: Data visualization

   Visualization of ``fractopo`` data. ``fractopo`` analyses the trace data
   that can e.g. be digitized from drone orthophotographs (fractures) or from
   digital elevation models (lineaments). The displayed branches and nodes are
   extracted with ``fractopo``.

Validating trace data
---------------------

Trace data must be validated using ``fractopo`` validation functionality before
analysis. The topological analysis of lineament & fracture traces implemented
in ``fractopo`` will not tolerate uncertainty related to the topological
abutting and snapping relationships between traces. Therefore the
trace validation is recommended before all analysis using ``Network``.

-  Introduction to trace data validation

   -  `Basics <validation/basics>`__
   -  `Error Types <validation/errors>`__

-  Basic trace data validation workflows

   -  `Notebook 1 <notebooks/fractopo_validation_1>`__
   -  `Notebook 2 <notebooks/fractopo_validation_2>`__

For trace validation also see `above guide <#trace-validation>`__ on
``tracevalidate`` command-line tool. After trace data validation, ``Network``
analysis can be conducted. Click on the links below for analysis examples
and guidance:

.. toctree::
   :maxdepth: 1
   :caption: Validating trace data
   :hidden:

   validation/basics
   validation/errors
   notebooks/fractopo_validation_1
   notebooks/fractopo_validation_2


Analyzing trace data
--------------------

This Python module can be used to extract lineament & fracture size, abundance
and topological parameters from two-dimensional lineament and fracture trace,
branch and node data.

It is recommended to `validate the trace data <validation/basics.html>`__ first
before using it as an input for extracting branches and nodes. Otherwise the
extraction process might error out or result in invalid branches and nodes.

-  `Workflow of analyzing trace data with visualizations of geometric
   topological network characteristics. Recommended for an introduction to
   fracture network analysis with fractopo. <notebooks/fractopo_network_1.html>`__
-  `Example scripts with resulting plots showcasing different analysis and
   plotting functionality implemented in fractopo. Recommended for overview of
   functionality. <auto_examples/index.html>`__

.. toctree::
   :maxdepth: 1
   :caption: Analyzing trace data
   :hidden:

   notebooks/fractopo_network_1

.. toctree::
   :maxdepth: 1
   :caption: Gallery
   :hidden:

   auto_examples/index

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

   -  Length distribution modelling using the Python 3 powerlaw
      package which ``fractopo`` uses
   -  `powerlaw GitHub <https://github.com/jeffalstott/powerlaw>`__

-  `Bonnet et al.,
   2001 <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/1999RG000074>`__

   -  Length distribution modelling review.

-  `My Master’s Thesis, Ovaskainen,
   2020 <http://urn.fi/URN:NBN:fi-fe202003259211>`__

   -  Plots used in my Thesis were done with an older version of the
      same code used for this plugin.

.. toctree::
   :maxdepth: 1
   :caption: Module documentation

   apidoc/fractopo


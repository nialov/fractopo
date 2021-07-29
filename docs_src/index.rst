.. include:: ../README.rst

Contents
========

Trace data must be validated using ``fractopo`` validation functionality before
analysis. The topological analysis of lineament & fracture traces implemented
in ``fractopo`` will not tolerate uncertainty related to the topological
abutting and snapping relationships between traces. Therefore the
trace validation is recommended before all analysis using ``Network``.



.. toctree::
   :maxdepth: 1
   :caption: Validating trace data

   validation/basics
   validation/errors
   notebooks/fractopo_validation_1
   notebooks/fractopo_validation_2

See above links for:

-  Basic trace data validation workflows
-  Examples of validation error types

For trace validation also see above guide on ``tracevalidate`` command-line
tool.  After trace data validation, ``Network`` analysis can be conducted.


.. toctree::
   :maxdepth: 1
   :caption: Analyzing trace data

   analysis
   notebooks/fractopo_network_1
   auto_examples/index

See above links for:

-  Workflow of analyzing trace data with visualizations of geometric
   topological network characteristics.
-  Example scripts showcasing different analysis and plotting functionality
   implemented in fractopo.
-  **Notebook - Fractopo – KB11 Fracture Network Analysis** is especially
   recommended.


.. toctree::
   :maxdepth: 1
   :caption: Module documentation for developers

   apidoc/fractopo


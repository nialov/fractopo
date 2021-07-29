Geometric and topological trace network analysis
================================================

Description
-----------

This Python module can be used to extract lineament & fracture size,
abundance and topological parameters from two-dimensional lineament and
fracture trace, branch and node data. The results will be given as
graphical plots.

Prerequisites
-------------

See: `Prerequisites <validation/basics.html#Prerequisites>`__

It is recommended to validate the trace data first before using it as an
input for extracting branches and nodes. Otherwise the extraction
process might error out or result in invalid branches and nodes.

Branches and nodes
------------------

Topologically a trace network of lineaments or fractures can be
dissolved into branches and nodes. Branches represent individual
segments of traces and each segment has a node on each end. Nodes
represent either interactions with other traces or isolated abutments.
See `Sanderson and Nixon,
2015 <https://www.sciencedirect.com/science/article/abs/pii/S0191814115000152?via%3Dihub>`__
for a more detailed explanation.

Usage
-----

Usage is demonstrated in complementary ``Jupyter Notebooks``.

-  `Network analysis <../notebooks/fractopo_network_1>`__

   -  Loading data for analysis with ``geopandas``
   -  Visualizing traces and target area
   -  Extracting branches and nodes
   -  Visualizing branches and nodes
   -  Azimuth rose plots
   -  Length distribution analysis
   -  Crosscutting and abutting relationships between azimuth sets
   -  Node and branch proportions

Also see `gallery of example scripts <../auto_examples/index.rst>`__

References
----------

For the definition of traces, branches and nodes along with the
explanation of the plots and the plotted parameters, I refer you to
multiple sources.

-  `Nyberg et al., 2018 <https://doi.org/10.1130/GES01595.1>`__

   -  *NetworkGT Plugin introduction and guide.*
   -  `NetworkGT GitHub <https://github.com/BjornNyberg/NetworkGT>`__

-  `Sanderson and Nixon,
   2015 <https://doi.org/10.1016/j.jsg.2015.01.005>`__

   -  *Trace and branch size, abundance and topological parameter
      definitions.*

-  `My Master’s Thesis, Ovaskainen,
   2020 <http://urn.fi/URN:NBN:fi-fe202003259211>`__

   -  *Plots used in my Thesis were done with an older version of the
      same code used for this plugin.*

-  `Sanderson and Peacock,
   2020 <https://www.sciencedirect.com/science/article/abs/pii/S001282521930594X>`__

   -  *Information about rose plots.*

-  `Alstott et al.
   2014 <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0085777>`__

   -  *Length distribution modelling using the Python 3 powerlaw
      package.*
   -  `powerlaw GitHub <https://github.com/jeffalstott/powerlaw>`__

-  `Bonnet et al.,
   2001 <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/1999RG000074>`__

   -  *Length distribution modelling.*

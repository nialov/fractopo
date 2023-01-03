---
title: 'fractopo: A Python package for working with fracture network data'
tags:
  - Python
  - geology
  - geographic information systems
authors:
  - name: Nikolas Ovaskainen
    orcid: 0000-0003-1562-0280
    <!-- equal-contrib: true -->
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Geological Survey of Finland
   index: 1
 - name: University of Turku, Finland
   index: 2
date: 9 December 2022
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# <!-- aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it. -->
# <!-- aas-journal: Astrophysical Journal <- The name of the AAS journal. -->
---

# Summary

In the Earth's crust, networks of planar discontinuities, fractures,
form intricate networks where they cross-cut and abut in each other.
These fractures control the stability and act as pathways for fluid flow
and subsequently, geothermal heat and contaminant transfer. The
fractures can be observed from exposed bedrock surfaces where the
roughly planar discontinuities cut the bedrock surface forming
two-dimensional fracture traces on the surface. Digitizing these
fracture traces from e.g., drone images of the outcrops results in
georeferenced two-dimensional fracture trace vector datasets, i.e.,
fracture networks, which offer a window into the three-dimensional
networks within the bedrock. To analyze these datasets, common
Geographic Information System (GIS) -tools can be used to perform
geospatial operations and analysis of the data. However, due to specific
requirements for geometric and topological consistency, it is convenient
for the user to have access to a common workflow for the data validation
and analysis, rather than implementing a geospatial analysis separately
each time for the data.

# Statement of need

The Python package, `fractopo`, provides the tooling for i) data validation and
ii) analysis of fracture network data (\autoref{fig:diagram}). The fracture
trace data is most commonly produced by manual digitization from base maps such
as drone images of bedrock outcrops. Consequently, a number of digitization
errors can occur in the data related to the geometric consistency of the traces
(e.g., traces must not be multi-part) or topological consistency (e.g., a trace
can be interpreted to abut to another trace only if it is within a certain
threshold distance from the other trace or if the endpoint of one is a point
along the other). To tackle the validation problem, `fractopo` uses the
`geopandas` [@jordahl_geopandasgeopandas_2022] and `shapely`
[@gillies_shapely_2022] Python packages to conduct a high variety of geometric
consistency checks to find topologically invalid traces that are the result of
common digitization errors which are then reported to the user as attributes of
the trace data.

For data analysis, `fractopo`  the user can simply input the trace data along
with its associated digitization boundary to `fractopo`. The package can then
be called to return a number of analysis results in the form of
`matplotlib`/`seaborn` [@hunter_matplotlib_2007; @waskom_seaborn_2021] plots
and as numerical data in the form of `numpy` arrays and `pandas` dataframes.
The results include e.g., rose plots of the orientation of the traces
[@sanderson_making_2020], power-law length distribution analysis of the lengths
of traces [@bonnet_scaling_2001], cross-cutting and abutting relationships
between predefined trace sets, fracture intensities [@sanderson_use_2015] and
topological ternary plots [@manzocchi_connectivity_2002; @sanderson_use_2015]
(\autoref{fig:montage}). The package bears much similarity, and is inspired by,
`NetworkGT` [@nyberg_networkgt_2018] which first provided a workflow for
analysis of fracture trace data, including the determination of topological
branches and nodes. However, the tight integration of `NetworkGT` with `QGIS`
causes the package to be less friendly to development as it restricts the use
of `NetworkGT` strictly inside `QGIS` (or alternatively `ArcGIS`). In contrast,
`fractopo`, can be used anywhere with either the `conda` or `pip` package
managers as well as containing features absent from `NetworkGT`, such as the
determination of cross-cutting relationships between groups of fractures.

![General workflow illustration of the data that `fractopo` takes and the
available results.\label{fig:diagram}](figs/fractopo_2d_diagram.png)

The usefulness of `fractopo` in research has been proven by its usage in two
publications [@skytta_fault-induced_2021; @ovaskainen_new_2022] as well as in
three Master's Theses [@ovaskainen_scalability_2020;
@jokiniemi_3d-modelling_2021; @lauraeus_3d-modelling_2021] and on a course,
*Brittle Structures in Bedrock; Engineering Geology* at the University of
Turku. Development continues actively and more academic publications are
in the works.

![Data analysis workflow visualisation for fracture trace data
(``KB11``). A. Target area for trace digitisation. B. Digitized
traces and target area. C. Orthomosaic used as the base raster from
which the traces are digitized from. D. Equal-area length-weighted
rose plot of the fracture trace azimuths. E. Length distribution
analysis of the trace lengths. F. Determined branches and nodes
through topological analysis. G. Cross-cut and abutting relationships
between chosen azimuth sets. H. Ternary plot of node (X, Y and I)
proportions. I. Ternary plot of branch (C-C, C-I, I-I) proportions.
](figs/fractopo_workflow_visualisation.jpg)

# Acknowledgements

This package has mainly been developed as part of a Finnish Research Programme
on Nuclear Waste Management (2019–2022) and Geological Survey of Finland funded
project, *KYT KARIKKO*. We acknowledge the funders for the opportunity to
develop this software.

# References

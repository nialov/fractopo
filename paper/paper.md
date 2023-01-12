---
title: 'fractopo: A Python package for fracture network analysis'
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

In the Earth's crust, networks of approximately planar discontinuities,
fractures, form intricate networks where they cross-cut and abut each other.
These fractures control the stability of the crust and act as pathways for
fluid flow and subsequently, transfer of geothermal heat and contaminants.
Fractures can be observed from exposed bedrock surfaces where these
discontinuities appear as two-dimensional fracture traces. Digitizing these
fracture traces from e.g. drone imaged outcrops results in georeferenced
two-dimensional fracture trace vector datasets, i.e. fracture networks, which
offer a cross-sectional window into the three-dimensional networks within the
bedrock. To analyze these datasets, common Geographic Information System (GIS)
tools are typically used to perform geospatial operations and analysis of the
data. However, these tools are not specialized to handle the specific
requirements for geometric and topological consistency of the fracture trace
data. Therefore, it is both convenient and beneficial for users to have
access to a common workflow, provided by the newly developed `fractopo`
package, for the data validation and analysis, rather than implementing a
geospatial analysis separately each time for the data.

# Statement of need

The Python package, `fractopo`, provides the tooling for i) data validation and
ii) analysis of fracture network data (\autoref{fig:diagram}). The fracture
trace data is most commonly produced from base maps (e.g. drone images of
outcrops or digital elevation models) by manual digitization. Consequently, a
number of digitization errors can occur in the trace data related to the
geometric consistency of the traces (e.g. traces must be continuous i.e.
without breaks between segments) or topological consistency (e.g. a trace can
be interpreted to abut another trace only if it is within a certain threshold
distance from the other trace or if the endpoint of one is a point along the
other). To tackle the validation problem, `fractopo` uses the `geopandas`
[@jordahl_geopandasgeopandas_2022] and `shapely` [@gillies_shapely_2022] Python
packages to conduct a high variety of geometric consistency checks to find
topologically invalid traces that are the result of common digitization errors
which are then reported to the user as attributes of the trace data.

For data analysis, the user can simply input the validated trace data along
with its associated digitization boundary to `fractopo` based on which a number
of analysis results in the form of `matplotlib`/`seaborn`
[@hunter_matplotlib_2007; @waskom_seaborn_2021] plots and as numerical data in
the form of `numpy` arrays and `pandas` dataframes can be generated. The
results include e.g. rose plots of the orientation of the traces
[@sanderson_making_2020], power-law length distribution analysis of the lengths
of traces [@bonnet_scaling_2001; @alstott_powerlaw_2014], cross-cutting and
abutting relationships between predefined trace sets, fracture intensities
[@sanderson_use_2015] and topological ternary plots
[@manzocchi_connectivity_2002; @sanderson_use_2015] (\autoref{fig:montage}).
The package bears much similarity, and is inspired by, `NetworkGT`
[@nyberg_networkgt_2018] which first provided a workflow for analysis of
fracture trace data, including the determination of topological branches and
nodes. However, the tight integration of `NetworkGT` with `QGIS` causes the
package to be less friendly to development as it restricts the use of
`NetworkGT` strictly inside `QGIS` (or alternatively `ArcGIS`, but with the now
deprecated version of `NetworkGT`). In contrast, `fractopo`, can be used
anywhere with either the `conda`, `pip` or `nix` package managers as well as
containing features absent from `NetworkGT`, such as the determination of
cross-cutting relationships between groups of fractures.

![General workflow illustration of the data that `fractopo` takes and the
available results.\label{fig:diagram}](figs/fractopo_2d_diagram.png)

Use of `fractopo` in research include two publications
[@skytta_fault-induced_2021; @ovaskainen_new_2022] as well as three
Master's Theses
[@ovaskainen_scalability_2020; @jokiniemi_3d-modelling_2021; @lauraeus_3d-modelling_2021]
and in assignments on a course, *Brittle Structures in Bedrock;
Engineering Geology* at the University of Turku. Development of
`fractopo` continues actively and the use of it continues in multiple
ongoing academic works.

![Visualisation of the workflow for fracture trace data analysis. A.
Target area for trace digitisation. B. Digitized traces and target area.
C. Orthomosaic used as the base raster from which the traces are
digitized from. D. Equal-area length-weighted rose plot of the fracture
trace azimuths. E. Length distribution analysis of the trace lengths. F.
Determined branches and nodes through topological analysis. G. Cross-cut
and abutting relationships between chosen azimuth sets. H. Ternary plot
of node (X, Y and I) proportions. I. Ternary plot of branch (C-C, C-I,
I-I) proportions.\label{fig:montage}](figs/fractopo_workflow_visualisation.jpg)

# Acknowledgements

This package has mainly been developed as part of a Finnish Research Programme
on Nuclear Waste Management (2019–2022) and Geological Survey of Finland funded
project, *KYT KARIKKO*. We acknowledge the funders for the opportunity to
develop this software.

# References

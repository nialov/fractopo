.. meta::
   :description: A comprehensive guide on digitizing geologcal fractures and lineaments using image (raster) data
   :keywords: data digitization, structural geology, GIS

How to digitize geological fractures and lineaments
===================================================

Introduction
------------

How to digitize fractures and lineaments using QGIS
---------------------------------------------------

This guide expects some basic knowledge of geographic information
systems (GIS) and their nomenclature. However, the guide is meant to be
as detailed as necessary for a beginner of QGIS to follow.

Setting up your project
~~~~~~~~~~~~~~~~~~~~~~~

Project directory
^^^^^^^^^^^^^^^^^

Before opening QGIS, set up an appropriate project directory for
digitization. Create a project directory, with a suitable name, e.g.
``brittle_course_fracture_digitization_2026`` in a suitable location
under your user directory. E.g. in
``C:\Users\<your-username>\projects\`` or any other locations you have
used for projects. Set up a suitable directory structure within that
project directory (``brittle_course_fracture_digitization_2026``) for
data. Make a ``data`` directory with subdirectories ``raster`` and
``vector``.

In the following text, I will refer to the project directory with the name ``brittle_course_fracture_digitization_2026`` and yours may differ.

Tree-view of the project folder:

.. code::

   brittle_course_fracture_digitization_2026
   └── data
       ├── raster
       └── vector

Now start up QGIS. Make a new project and save the project file in ``brittle_course_fracture_digitization_2026``. I would recommend
using the same name for the project file as you have for the project
directory. After saving the project file in the directory, you should
have a new ``brittle_course_fracture_digitization_2026.qgz`` file there.

.. code::

   brittle_course_fracture_digitization_2026
   ├── brittle_course_fracture_digitzation_2026.qgz
   └── data
       ├── raster
       └── vector

Project settings
^^^^^^^^^^^^^^^^

The project coordinate reference system (CRS) should be set to a metric coordinate system
to avoid confusing results from analysis of the finished digitized traces.
If you are in Finland, the recommended CRS is EPSG:3067 (EUREF-FIN / TM35FIN(E,N) - Finland).

To check and set it, go to TODO

The topological editing tool needs to be added to the toolbar
for easy access. TODO

Adding data to the project
~~~~~~~~~~~~~~~~~~~~~~~~~~

Raster data
^^^^^^^^^^^

Moving the raster data you are going to use to the ``data/rasters/``
folder of the project directory is recommended as the data will then be
easily accessible in QGIS.

.. note::

   If you use the same raster data in multiple projects, it might be
   better to store it in a central location rather than copying it to
   each individual project folder, as the raster data itself is never
   edited during digitization.

To add raster data to QGIS go to TODO

Check that the added raster data uses the same CRS as the project.

Vector data
^^^^^^^^^^^

The rasters can cover large areas and digitizing the whole extent might
not be needed. Consequently, it is a good idea to create a preliminary
target area for digitizing at this point.

Create a new polygon vector layer for the target area by TODO

Create a new ``LineString`` layer for the traces to be digitized.
At its simpltest, the trace layer can only consist of the trace geometries
without any attribute information.

How to digitize fractures and lineaments using ArcGIS Pro
---------------------------------------------------------

How to collect metadata for digitized fractures
-----------------------------------------------

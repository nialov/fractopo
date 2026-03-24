.. meta::
   :description: A comprehensive guide on digitizing geologcal fractures and lineaments using image (raster) data
   :keywords: data digitization, structural geology, GIS

How to digitize geological fractures and lineaments
===================================================

Introduction
---------------

Though drawing lines on a map might not seem complex, there are still
rules to follow to make the data you produce is analyzable without
inconsistencies.

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
created project directory for
data by creating a ``data`` directory with two subdirectories: ``raster`` and
``vector``.

In the following text, I will refer to the project directory with the
name ``brittle_course_fracture_digitization_2026`` and yours may differ.

Tree-view of the project folder:

.. code::

   brittle_course_fracture_digitization_2026
   └── data
       ├── raster
       └── vector

Now start up QGIS. Make a new project and save the project file in
``brittle_course_fracture_digitization_2026``. I would recommend using
the same name for the project file as you have for the project
directory. After saving the project file in the directory, you should
have a new ``brittle_course_fracture_digitization_2026.qgz`` file there.

.. code::

   brittle_course_fracture_digitization_2026
   ├── brittle_course_fracture_digitization_2026.qgz
   └── data
       ├── raster
       └── vector

Project settings
^^^^^^^^^^^^^^^^

The project coordinate reference system (CRS) should be set to a metric coordinate system
to avoid confusing results from analysis of the finished digitized traces.
If you are in Finland, the recommended CRS is EPSG:3067 (EUREF-FIN / TM35FIN(E,N) - Finland).

To check and set it, go to ``Project`` -> ``Properties`` -> ``CRS``. Use
``Filter`` to search for ``EPSG:3065`` and select it. Then click ``OK``
to save the setting.

.. dropdown::  Screenshot of CRS selection screen
   :animate: fade-in

   .. figure:: screenshots/qgis_project_crs.jpg
      :alt: Screenshot of CRS selection screen

   Project coordinate selection window with EPSG:3067 selected.

The topological editing tool needs to be added to the toolbar for easy
access. Go to ``View`` -> ``Toolbars`` and toggle ``Snapping toolbar``
on.

.. dropdown:: Screenshot of enabling the QGIS snapping toolbar
   :animate: fade-in

   .. figure:: screenshots/qgis_snapping_toolbar.jpg
      :alt: Screenshot of enabling the QGIS snapping toolbar

   Toggle the ``Snapping toolbar`` on, if it is not already. In the
   image it is toggled on. It should appear where the upper red
   rectangle shows (or within the toolbar somewhere else).

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

To add raster data into QGIS, go to ``Layer`` -> ``Add Layer`` ->
``Add Raster Layer`` and select your raster file, which usually has a
``.tif`` extension, and click ``Add`` to add it to the project.

.. dropdown:: Adding raster data in QGIS
   :animate: fade-in

   .. figure:: screenshots/qgis_add_raster_layer_1.jpg
      :alt: Adding raster data in QGIS

      After choosing the raster file, click ``Add`` to add it to the project.

Check that the added raster data uses the same CRS as the project. Right
click on the layer in the ``Layers`` tab, click ``Properties``. In the
``Layer Properties`` window, go to ``Source`` and check the coordinate
system and change it to the project coordinate system, if necessary.

.. dropdown:: Screenshot of checking raster layer settings for CRS
   :animate: fade-in

   .. figure:: screenshots/qgis_check_raster_layer_crs.jpg
      :alt: Screenshot of checking raster layer settings for CRS

      Check the CRS in the area indicated by red rectangle.

Vector data
^^^^^^^^^^^

The rasters can cover large areas and digitizing the whole extent might
not be needed. Consequently, it is a good idea to create a preliminary
target area for digitizing at this point.

If you do not have an existing target area, create a new polygon vector
layer for the target area by going to ``Layer`` -> ``Create layer`` ->
``New GeoPackage Layer``.

.. dropdown:: Screenshot of navigating to ``New GeoPackage Layer`` option
   :animate: fade-in

    .. figure:: screenshots/qgis_add_vector_layer.jpg
       :alt: Screenshot of navigating to ``New GeoPackage Layer`` option

       Usage of GeoPackages for storing vector data is recommended due to their
       high compatibility and single-file database structure.

.. dropdown:: Screenshot of vector layer creation screen with options selected for creating a GeoPackage file for storing target are Polygon data.
   :animate: fade-in

   .. figure:: screenshots/qgis_add_vector_area_layer.jpg
      :alt: screenshots/qgis_add_vector_area_layer.jpg

      Please carefully check 1. that the ``Table name`` is the same as
      the filename, without the extension (``.gpkg``) as seen in
      ``File name``, 2. ``Geometry type`` is ``Polygon`` and 3. the CRS
      is the same as the project.


Create a new ``LineString`` layer for the traces to be digitized.
At its simplest, the trace layer can only consist of the trace geometries
without any attribute information. When creating your trace layer, select **LineString** as the geometry type.

.. note::

   Avoid using MultiLineString geometry type. If your lines are
   accidentally stored as MultiLineStrings, use QGIS’s “Explode Lines”
   tool (Processing Toolbox > Vector geometry > Explode lines) to
   convert them to individual LineStrings.

Digitizing fracture and lineament traces in QGIS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Enable and configure snapping**

-  Click the magnet icon in the QGIS toolbar or go to Project > Snapping
   Options.
-  Set snapping to “Vertex” and “Segment” for your trace layer, and
   choose a small tolerance (e.g., TODO).
-  This helps ensure that line endpoints connect precisely, which is
   important for accurate network analysis, i.e., determining which
   trace abuts another and which crosscuts.

**2. Avoid unintended intersections**

-  Do not let more than two lines intersect at a single point. If
   multiple lines cross at one spot, edit them so only two intersect.
-  When two lines are meant to connect, make sure their endpoints are
   snapped together. Use the “Vertex Tool” to adjust endpoints as
   needed.

**3. Prevent self-intersections and duplicate lines**

-  Make sure each line does not cross itself. Use the “Check Geometry
   Validity” tool (Processing Toolbox > Vector geometry > Check
   validity) to identify and fix self-intersections.
-  Avoid drawing duplicate lines directly on top of each other. If you
   find duplicates, delete them.

**4. Trace length and target area**

-  If you have created a target area to control where you are going to
   digitize, make sure you do not stop your traces at the boundary.
   Rather, continue them outside the boundary as far as they can be
   interpreted to continue. Otherwise trace lengths might be improperly
   samples. Furthermore, the target area you currently have might be
   extended in the future.

How to digitize fractures and lineaments using ArcGIS Pro
---------------------------------------------------------

How to collect metadata for digitized fractures
-----------------------------------------------

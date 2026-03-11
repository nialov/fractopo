.. meta::
   :description: A comprehensive guide on digitizing geologcal fractures and lineaments using image (raster) data
   :keywords: data digitization, structural geology, GIS

How to digitize geological fractures and lineaments
===================================================

Introduction
------------

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
project directory (``brittle_course_fracture_digitization_2026``) for
data. Make a ``data`` directory with subdirectories ``raster`` and
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

To add raster data in QGIS go to TODO

Check that the added raster data uses the same CRS as the project.

Vector data
^^^^^^^^^^^

The rasters can cover large areas and digitizing the whole extent might
not be needed. Consequently, it is a good idea to create a preliminary
target area for digitizing at this point.

Create a new polygon vector layer for the target area by TODO

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

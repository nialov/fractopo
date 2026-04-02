.. meta::
   :description: A comprehensive guide on digitizing geologcal fractures and lineaments using image (raster) data
   :keywords: data digitization, structural geology, GIS

How to digitize geological fractures and lineaments
===================================================

Introduction
---------------

Though drawing lines on a map might not seem complex, there are still
rules to follow to make the data you produce analyzable without
inconsistencies.

Digitizing lineaments and fractures follows the same process. Usually the
scale of observation and underlying raster data are different but actual
process of digitization is the same. Consequently, any specific
references to "fractures" or "lineaments" are mostly interchangeable,
unless otherwise specified.

The purpose of digitizing is usually creating digital data about
bedrock discontinuities, i.e. fractures and faults. By drawing
along a fracture on a georeferenced picture of an outcrop,
you are documenting the length and orientation of a bedrock feature.
Furthermore, by accurately digitizing relationships between fractures,
you are producing topological information about how fractures
interact and form a network.

.. dropdown::  Examples and illustrations of fracture digitizing
   :animate: fade-in

   .. figure:: figures/zoomed_digitization_example.jpg
      :alt: Image showing outcrop photo on the left side and digitized fractures on top of the outcrop photo on the right side

   On the left: outcrop photo, on the right: outcrop photo with digitized fractures on top of it

   .. figure:: figures/orthomosaic_with_fractures_and_inferred_fault.jpg
      :alt: Image showing outcrop photo with thousands of digitized fractures on top and an E-W trending annotated fault

   Drone orthomosaic from the northern shores of Åland Islands
   with thousands of digitized fractures on top and an inferred
   fault zone, i.e. a remotely interpreted potential fault.

How to digitize fractures and lineaments using QGIS
---------------------------------------------------

This guide expects some basic knowledge of geographic information
systems (GIS) and their nomenclature. However, the guide is meant to be
as detailed as necessary for a beginner of QGIS to follow.

This guide has been written for QGIS version 4.0. Some deviation in
terms of different option names and locations is therefore possible
if you are using a different version.

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

.. code:: text

   brittle_course_fracture_digitization_2026
   └── data
       ├── raster
       └── vector

Now start up QGIS. Make a new project and save the project file in
``brittle_course_fracture_digitization_2026``. I would recommend using
the same name for the project file as you have for the project
directory. After saving the project file in the directory, you should
have a new ``brittle_course_fracture_digitization_2026.qgz`` file there.

.. code:: text

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
``.tif`` or ``.tiff`` extension, and click ``Add`` to add it to the project.

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

.. dropdown:: Screenshot of vector layer creation screen with options selected for creating a GeoPackage file for storing target area Polygon data.
   :animate: fade-in

   .. figure:: screenshots/qgis_add_vector_area_layer.jpg
      :alt: Screenshot of vector layer creation screen with options selected for creating a GeoPackage file for storing target area Polygon data.

      Please carefully check 1. that the ``Table name`` is the same as
      the filename, without the extension (``.gpkg``) as seen in
      ``File name``, 2. ``Geometry type`` is ``Polygon`` and 3. the CRS
      is the same as the project.

Create a new ``LineString`` layer for the traces to be digitized.
At its simplest, the trace layer can only consist of the trace geometries
without any attribute information. When creating your trace layer, select **LineString** as the geometry type.

.. dropdown:: Screenshot of vector layer creation screen with options selected for creating a GeoPackage file for storing ``LineString`` trace data.
   :animate: fade-in

   .. figure:: screenshots/qgis_add_vector_traces_layer.jpg
      :alt: Screenshot of vector layer creation screen with options selected for creating a GeoPackage file for storing ``LineString`` trace data.

      Please carefully check 1. that the ``Table name`` is the same as
      the filename, without the extension (``.gpkg``) as seen in
      ``File name``, 2. ``Geometry type`` is ``LineString`` and 3. the CRS
      is the same as the project.

.. note::

   Avoid using MultiLineString geometry type. If your lines are
   accidentally stored as MultiLineStrings, use QGIS’s “Explode Lines”
   tool (Processing Toolbox > Vector geometry > Explode lines) to
   convert them to individual LineStrings.

Digitizing fracture and lineament traces in QGIS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Layer organization
^^^^^^^^^^^^^^^^^^

QGIS determines what gets shown on top of what layer in the ordering of
layers in the ``Layers`` tab. Make sure the raster layer is at the
bottom, target area layer is on top of it and the layer with the traces
you are digitizing is at the top.

.. dropdown:: Screenshot of Layers tab with layers organized
   :animate: fade-in

   .. figure:: screenshots/qgis_layer_organization.jpg
      :alt: Screenshot

   Order of target area and traces is not so important.

Raster layer styling
^^^^^^^^^^^^^^^^^^^^

Orthomosaics, or pictures of outcrops in general, are RGB images
where additional styling is usually not required. However, when
digitizing lineaments using, e.g., digital elevation model (DEM)
data, you might need to configure styling.

Vector layer styling
^^^^^^^^^^^^^^^^^^^^^^

The target area is defined by polygon(s). By default, QGIS styles them
with a single color fill, i.e., they mask the layers beneath them.
You can change the styling in the ``Symbology`` section, accessed by right
clicking the layer in the ``Layers`` tab and selecting ``Properties``.

.. dropdown:: Setting style for target area layer
   :animate: fade-in

   .. figure:: screenshots/qgis_target_area_style.jpg
      :alt: Screenshot

   Click ``Symbology``, then, if ``Simple Fill`` (or another fill type)
   is being currently used, click it. Then change the
   ``Symbol layer type`` to ``Outline: Simple Line`` to only show the
   boundary of the polygon.

To make digitizing easier, you can try adjusting the trace styles.
Changing the color is particularly helpful for ensuring your work stands
out against the underlying raster layer.

Enable and configure snapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Click the magnet icon in the QGIS toolbar or go to Project > Snapping
Options. Set snapping to “Vertex” and “Segment” for your trace layer,
and choose a small snapping tolerance (e.g., 12 ``px``). Snapping helps
to ensure that traces precisely abut, i.e. endpoint of one trace is
exactly along a segment or on top of a vertex of another trace, to other
traces, which is important for accurate topological network analysis.

.. dropdown:: Configuring snapping in the toolbar
   :animate: fade-in

   .. figure:: screenshots/qgis_configure_snapping_1.jpg
      :alt: Screenshot

   Toggle  ``Enable Snapping`` and set snapping to ``Vertex`` and ``Segment``.

   .. figure:: screenshots/qgis_configure_snapping_2.jpg
      :alt: Screenshot

   1. Configure that new features snap to ``Active Layer`` so that when
   you create new features, they only snap to features in the same
   layer. 2. Set ``Snapping Tolerance`` to 12 ``px``. This determines
   how far the cursor will try to snap to old features when creating new
   ones.

Digitizing new traces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before starting to digitize traces, you set disable the pop-up that
appears for inputting attribute data after each feature is digitized.
Go to ``Settings`` -> ``Options`` -> ``Map Tools`` -> ``Digitizing``.
Check ``Suppress attribute form ...`` to disable the pop-up.

.. dropdown:: Suppressing attribute form
   :animate: fade-in

   .. figure:: screenshots/qgis_digitize_features_option.jpg
      :alt: Screenshot

   Toggle ``Suppress attribute form ...`` on to disable the pop-up.

To start digitizing new features, make sure you have selected the trace layer
in the ``Layers`` tab. Next, put ``Toggle Editing`` toggle on. To create a new
feature, click on ``Add Line Feature`` button. Click on the map to create a vertex,
then click again to connect the vertices, and continue digitizing from
one end of the lineament or fracture to the other end. If either end of the fracture
seems to abut another fracture

.. dropdown:: Digitizing new features
   :animate: fade-in

   .. figure:: screenshots/qgis_digitize_features_1.jpg
      :alt: Screenshot

   Toggle editing (pen icon) on, then click the create feature button.
   Make sure that snapping is turned on (magnet icon).

   .. figure:: screenshots/qgis_digitize_features_2.jpg
      :alt: Screenshot

   Remember to save layer changes. You can check if you have unsaved
   changes from the ``Layers`` tab or from the save button itself.

Modifying existing traces
^^^^^^^^^^^^^^^^^^^^^^^^^

It is often necessary to modify already digitized features due to
reinterpretation and to match them with other digitized fractures.
Start editing similar to when creating new traces. Instead of clicking
``Add Line Feature``, click on ``Vertex Tool``. Now when moving your
cursor above traces, you should see their vertices highlighted.
To modify a single vertex, click on it. You can then move it and
the trace will be modified to fit the new vertex. To add a new
vertex between two existing vertices, click along the trace somewhere
where there is no vertex between the two vertices. To continue a trace,
click on the plus-symbol at either end of the trace to start appending
vertices.

.. note::

   When editing traces, you might accidentally cause another trace to no
   longer abut the modified trace. You can avoid this by adding a
   vertice along the modified trace at the endpoint of fracture abutting
   the modified trace.

.. dropdown:: Modifying existing features
   :animate: fade-in

   .. figure:: screenshots/qgis_modify_features.jpg
      :alt: Screenshot

   Toggle editing (pen icon) on, then click the vertex button.
   Make sure that snapping is turned on (magnet icon) also when editing.

How to digitize fractures and lineaments using ArcGIS Pro
---------------------------------------------------------

Things to keep in mind while digitizing
---------------------------------------

-  Avoid unintended intersections

   -  Do not let more than two lines intersect at a single point. If
      multiple lines cross at one spot, edit them so only two intersect.
   -  When two lines are meant to connect, make sure their endpoints are
      snapped together. Use the “Vertex Tool” to adjust endpoints as
      needed.

-  Prevent self-intersections and duplicate lines

   -  Make sure each line does not cross itself. Use the “Check Geometry
      Validity” tool (Processing Toolbox > Vector geometry > Check
      validity) to identify and fix self-intersections.
   -  Avoid drawing duplicate lines directly on top of each other. If
      you find duplicates, delete them.

-  Trace length and target area

   -  If you have created a target area to control where you are going
      to digitize, make sure you do not stop your traces at the
      boundary. Rather, continue them outside the boundary as far as
      they can be interpreted to continue. Otherwise trace lengths might
      be improperly samples. Furthermore, the target area you currently
      have might be extended in the future.

For more detailed instructions regarding the different digitization
errors, go to the :doc:`/validation/errors` page.

How to collect metadata for digitized fractures
-----------------------------------------------

Next steps
----------

If you want to validate your data using ``fractopo``, you can do so
using the command-line interface (See :doc:`/index`), using Python
code in a script or a notebook (See
:doc:`/notebooks/fractopo_validation_1`) or using the validation web interface if you have it available (See :ref:`guides/how_to_use_fractopo_web_interface:Validation`).

Similarly, to analyze data using ``fractopo``, you can use the
command-line interface, follow a notebook
(:doc:`/notebooks/fractopo_network_1`) or use the web
interface (See :ref:`guides/how_to_use_fractopo_web_interface:network analysis`).

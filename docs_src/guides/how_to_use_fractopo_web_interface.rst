.. meta::
   :description: A guide on using the web validation and network analysis interface of fractopo
   :keywords: data validation, data analysis, structural geology, marimo

How to use ``fractopo`` web interfaces for validation and network analysis
==========================================================================

A web interface for validation and network analysis of fractures and
lineaments are provided as part of ``fractopo``. See :doc:`/index` for
info about getting it running. You might have it "privately" available
as part of a course or project. Please use the link(s) provided to you
to access the interface. Validation and network analysis are available
under different links.

The interfaces are meant to be as intuitive as possible. Input data
consists of a geospatial database file containing the fracture or
lineament trace data, e.g. a GeoPackage (``.gpkg``) file and another
database file with the target area. Use the two upload areas to upload
both the trace and area databases. You need to specify the layer name
for the right layer in the database **if it does not match the filename
of the database**. E.g. if your fracture traces are in ``traces.gpkg``
with layer name ``traces``, you do not need to specify the layer name.

After uploading these two data, you might need to change the settings of
the process before running validation or analysis. See
:ref:`guides/how_to_use_fractopo_web_interface:validation` and :ref:`guides/how_to_use_fractopo_web_interface:network analysis`
subsections below.

Validation
----------

The validation interface should look something like this:

.. dropdown::  Screenshot of validation web interface
   :animate: fade-in

   .. figure:: screenshots/validation_web_interface.jpg
      :alt: Screenshot of validation web interface

   Validation web interface of ``fractopo``.

You normally do not need to change any settings before running
validation. If you run into problems, you might want to enable verbose
output and report the problem as an ``Issue`` on GitHub:
https://github.com/nialov/fractopo/issues. Note that validation errors
(:doc:`/validation/errors`) are something you need to fix in the trace
(or area) data, not a problem of the interface.

Network analysis
----------------

The network analysis interface should look something like this:

.. dropdown::  Screenshot of analysis web interface
   :animate: fade-in

   .. figure:: screenshots/analysis_web_interface.jpg
      :alt: Screenshot of analysis web interface

   Network analysis web interface of ``fractopo``.

You usually want to at least check the contour grid cell size,
``Name for analysis`` and ``Is the target area a circle?`` settings.
Contour grid cell size determines the size of the rectangles used to
sample the whole target area for parameters, such as, fracture
intensity. Name for the analysis determines the name used in e.g. plot
titles that are outputted from the analysis. If the used target area is
shaped like a circle, this can be used to provide additional statistical
parameters and, e.g, handle partly unknown trace lengths better. To get
more info about some of the settings, click the link that is embedded in
the option title text.

If you want to set azimuth sets (often called fracture sets), click the
``Define azimuth sets?`` toggle. Next, set the number of wanted sets.
Then, for each set, input the minimum and maximum value for the set.
Note that you can define a set that wraps around 0. E.g. a set,
labeled as ``N-S`` could be defined to start from 165 and end
at 15. See screenshots below.

.. dropdown:: Examples of defining azimuth sets
   :animate: fade-in

   .. figure:: screenshots/analysis_web_interface_azimuth_sets_simple.jpg
      :alt: Screenshot

   Example of definition of three azimuth sets with automatically
   defined labels.

   .. figure:: screenshots/analysis_web_interface_azimuth_sets_complex.jpg
      :alt: Screenshot

   Example definition of two azimuth sets where the first (``N-S``)
   wraps around 0 degrees and with user-defined labels.

Downloading and exploring output data
-------------------------------------

.. note::

   Both validation and network analysis take some processing time before
   the results appear. Wait patiently for the results to show and scroll
   to the bottom to find the download link. For few hundreds of
   digitized traces, the computing time is usually in seconds or tens of
   seconds. However, if you define a contour grid cell size, the
   processing time will be significantly increased due to the intensity
   of contouring.

Both the validation and analysis interfaces output the results as a
downloadable ``.zip`` archive. Some textual results are also shown
during processing. Extract the results from the archive to explore them.
The results, including databases, plots and textual data, are provided
in multiple formats for maximum interoperability. The file name
identifies the contents. You can use the file type most suitable to you.

Validation exports
~~~~~~~~~~~~~~~~~~

Validation results consists of just the traces in few different
geospatial file formats. If you open the downloaded traces, you will
notice the inclusion of a new column, ``VALIDATION_ERRORS``, within the
attribute table of the data. Use the values there to identify errors in
the trace data and fix them accordingly. You can fix them in the
original data you inputted or directly in the output trace data.
After fixing your trace data, run it again through the interface.
Make sure to use the file where you have done the edits.

As a general tip, make sure to clearly fix the errors pointed out by the
validation to avoid having to do multiple rounds of validation. For
example, if multiple traces intersect near each other, edit them to
clearly avoid each other.

.. note::

   If the trace data you inputted contained additional attribute data,
   make sure to check that it is correct in the output data as well.
   Data transformations during ``fractopo`` processing might cause
   unexpected changes in attribute columns or values.

Network analysis exports
~~~~~~~~~~~~~~~~~~~~~~~~

The network analysis exports include a selection of most basic analysis
that can be done on a fracture/lineament network. The exports also
include the defined branches and nodes which can be further explored or
analyzed outside ``fractopo``. Use the file names to identify what the
files contain. Most often used plots results include orientation plots
(e.g. ``trace_length_weighted_rose_plot.png``), contour plots (e.g.
``P21_contour.png``) and length distribution plots (e.g.
``trace_length_distribution_fits.png``).

.. note::

   The exports only contain a selected set of analysis possible with
   ``fractopo``. Further analysis is possible by calling ``fractopo`` as
   a Python library i.e. by coding (See
   :doc:`/notebooks/fractopo_network_1` and :doc:`/auto_examples/index`).

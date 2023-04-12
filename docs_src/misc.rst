Miscellanous/FAQ
================

Geometries with Z-coordinates
-----------------------------

Currently ``fractopo`` has only partly been tested with geometries that
have Z-coordinates (e.g. ``LineString([(0, 0, 0), (1, 1, 1)])``. Any
issues you believe might be related to Z-coordinates should be reported
on *GitHub*. To test for such issues using your data, try removing the
Z-coordinates with e.g. the following code:

   .. code:: python

      from shapely.geometry import Polygon, MultiPolygon, LineString


      def remove_z_coordinates(geometries):
          """
          :param geometries: Shapely geometry iterable
          :return: a list of Shapely geometries without z coordinates
          """
          return [
              shapely.wkb.loads(shapely.wkb.dumps(geometry, output_dimension=2))
              for geometry in geometries
          ]


      trace_data.geometry = remove_z_coordinates(trace_data.geometry)

Thanks to *Siyh* for the issue report and above code example
(https://github.com/nialov/fractopo/issues/21).

Furthermore, some validation and analysis remove Z-coordinates from
geometric results. Specifically, use of ``Validation`` will remove
Z-coordinates to avoid unexpected behaviour. Use of ``Network`` will by
default remove z-coordinates but this can be disabled (See ``Network``
input arguments)..

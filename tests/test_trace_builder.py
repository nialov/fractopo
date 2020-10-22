import shapely
import geopandas as gpd

from fractopo.tval import trace_builder


def test_line_generator():
    ps = [
        shapely.geometry.Point(0, 0),
        shapely.geometry.Point(1, 1),
    ]
    result = trace_builder.line_generator(ps)
    assert isinstance(result, shapely.geometry.LineString)
    assert result.length > 1


def test_multi_line_generator():
    ps1 = [
        shapely.geometry.Point(0, 0),
        shapely.geometry.Point(1, 1),
    ]
    ps2 = [
        shapely.geometry.Point(4, 3),
        shapely.geometry.Point(3, 2),
    ]
    result = trace_builder.multi_line_generator((ps1, ps2))
    assert isinstance(result, shapely.geometry.MultiLineString)
    # assert result.length > 1


def test_main():
    (
        valid_geoseries,
        invalid_geoseries,
        valid_areas_geoseries,
        invalid_areas_geoseries,
    ) = trace_builder.main()

    assert all(
        [
            isinstance(geosrs, gpd.GeoSeries)
            for geosrs in (
                valid_geoseries,
                invalid_geoseries,
                valid_areas_geoseries,
                invalid_areas_geoseries,
            )
        ]
    )
    assert all([geom.length > 0.001 for geom in valid_geoseries])
    assert all([geom.length > 0.001 for geom in invalid_geoseries])

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Final, Any, Type, Union
from datetime import datetime
from pathlib import Path
import time
import logging

import geopandas as gpd
from shapely.geometry import Point, LineString
from geopandas.sindex import PyGEOSSTRTreeIndex

from fractopo.tval import trace_validator
from fractopo.tval.trace_validation import ALL_VALIDATORS

from fractopo.general import determine_general_nodes

logging.basicConfig(
    level=logging.WARNING, format="%(process)d-%(levelname)s-%(message)s"
)


def main(
    traces: List[gpd.GeoDataFrame],
    areas: List[gpd.GeoDataFrame],
    names: List[str],
    auto_fix: bool,
) -> Dict[str, gpd.GeoDataFrame]:
    """
    Main entrypoint for validating a list of trace GeoDataFrames accompanied
    with a list of respective area GeoDataFrames and a list of names.

    Returns a dictionary of validated trace GeoDataFrames with names as keys.
    """
    gdfs = dict()
    for trace, area, name in zip(traces, areas, names):
        logging.info(f"Validating {name}")
        assert isinstance(trace, gpd.GeoDataFrame)
        assert isinstance(area, gpd.GeoDataFrame)
        if trace_validator.BaseValidator.ERROR_COLUMN in trace.columns:
            print(f"Error column already in gdf. Old will be backed up.")
            trace[f"error_column_backup"] = trace[
                trace_validator.BaseValidator.ERROR_COLUMN
            ]
            trace[trace_validator.BaseValidator.ERROR_COLUMN] = [
                [] for _ in trace.index.values
            ]
        for validator in (
            trace_validator.GeomNullValidator,
            trace_validator.EmptyGeometryValidator,
            trace_validator.SimpleGeometryValidator,
            trace_validator.GeomTypeValidator,
            trace_validator.MultiJunctionValidator,
            trace_validator.VNodeValidator,
            trace_validator.MultipleCrosscutValidator,
            trace_validator.UnderlappingSnapValidator,
            trace_validator.TargetAreaSnapValidator,
            trace_validator.StackedTracesValidator,
            trace_validator.SharpCornerValidator,
        ):
            logging.info(f"Validating with {validator}")
            logging.info(f"Start time is {time.asctime()}")
            trace = validator.execute(trace, area, auto_fix=auto_fix, parallel=True)
        gdfs[str(name)] = trace.copy()
        # Empty all previous node calculations when new layer is processed.
        trace_validator.BaseValidator.empty_node_data()
    return gdfs


def save_validated_gdfs(gdfs: Dict[str, gpd.GeoDataFrame], savefolder: Path):
    assert isinstance(gdfs, dict)
    for key in gdfs:
        assert isinstance(key, str)
        assert isinstance(savefolder, Path)
        gdf = gdfs[key]
        assert isinstance(gdf, gpd.GeoDataFrame)
        day = datetime.now().strftime("%d")
        month = datetime.now().strftime("%m")
        year = datetime.now().strftime("%Y")
        if not savefolder.exists() or savefolder.is_file():
            logging.info(
                f"Savefolder not found. Making directory with name: {savefolder}"
            )
            savefolder.mkdir()
        savepath = savefolder / Path(f"{key}_validated_{day}{month}{year}.gpkg")
        if savepath.exists():
            logging.warning(f"Savepath for {key} already exists. Overwriting.")

        # Transform ERROR_COLUMN column type from list to string
        gdf = gdf.astype({trace_validator.BaseValidator.ERROR_COLUMN: "str"})

        logging.info(f"Saving {key} to {savepath}.")
        gdf.to_file(savepath, driver="GPKG")


# def main_threaded(traces: List[Path], areas: List[Path]) -> Dict[str, gpd.GeoDataFrame]:
#     """
#     Run executions in threads when able. This results in a slight performance
#     increase over main.

#     The debugging of errors is however more difficult. If execution mysteriously
#     fails, I recommend using the main function to see if errors are raised.
#     """
#     gdfs = dict()
#     for trace, area in zip(traces, areas):
#         # print(f"Validating {trace}")
#         trace_gdf = gpd.read_file(trace)
#         if trace_validator.BaseValidator.ERROR_COLUMN in trace_gdf.columns:
#             print(f"Error column already in gdf. Old will be backed up.")
#             trace_gdf[f"error_column_backup"] = trace_gdf[
#                 trace_validator.BaseValidator.ERROR_COLUMN
#             ]
#             trace_gdf[trace_validator.BaseValidator.ERROR_COLUMN] = [
#                 [] for _ in trace_gdf.index
#             ]
#         area_gdf = gpd.read_file(area)
#         with ThreadPoolExecutor(max_workers=4) as pool_executor:
#             execute_threads = []
#             execute_names = []
#             for validator in (
#                 trace_validator.GeomNullValidator,
#                 trace_validator.GeomTypeValidator,
#                 trace_validator.MultiJunctionValidator,
#                 trace_validator.VNodeValidator,
#                 trace_validator.MultipleCrosscutValidator,
#                 trace_validator.UnderlappingSnapValidator,
#                 trace_validator.TargetAreaSnapValidator,
#             ):
#                 # print(f"Validating with {validator}")
#                 # print(f"Start time is {time.asctime()}")
#                 if validator in [
#                     trace_validator.GeomTypeValidator,
#                     trace_validator.GeomNullValidator,
#                     trace_validator.MultiJunctionValidator,
#                 ]:
#                     trace_gdf = validator.execute(
#                         trace_gdf, area_gdf, auto_fix=True, parallel=True
#                     )
#                 else:
#                     execute_threads.append(
#                         pool_executor.submit(
#                             validator.execute, *(trace_gdf, area_gdf, True, True)
#                         )
#                     )
#                     execute_names.append(str(trace))
#             error_columns = []
#             for future, name in zip(as_completed(execute_threads), execute_names):
#                 error_columns.append(future.result()[BaseValidator.ERROR_COLUMN])

#         trace_gdf[BaseValidator.ERROR_COLUMN] = assemble_error_column(
#             trace_gdf[BaseValidator.ERROR_COLUMN], error_columns
#         )
#         gdfs[str(trace)] = trace_gdf

#     return gdfs


# # def main_multiprocessing(
# #     traces: List[Path], areas: List[Path]
# # ) -> Dict[str, gpd.GeoDataFrame]:
# #     """
# #     Run executions in multiprocessing when able.
# #     """
# #     gdfs = dict()
# #     for trace, area in zip(traces, areas):
# #         print(f"Validating {traces}")
# #         trace_gdf = gpd.read_file(traces)
# #         if trace_validator.BaseValidator.ERROR_COLUMN in trace_gdf.columns:
# #             print(f"Error column already in gdf. Old will be backed up.")
# #             trace_gdf[f"error_column_backup"] = trace_gdf[
# #                 trace_validator.BaseValidator.ERROR_COLUMN
# #             ]
# #             trace_gdf[trace_validator.BaseValidator.ERROR_COLUMN] = [
# #                 [] for _ in trace_gdf.index
# #             ]
# #         area_gdf = gpd.read_file(area)
# #         with ProcessPoolExecutor(max_workers=2) as pool_executor:
# #             execute_processes = []
# #             execute_names = []
# #             for validator in (
# #                 trace_validator.GeomNullValidator,
# #                 trace_validator.GeomTypeValidator,
# #                 trace_validator.MultiJunctionValidator,
# #                 trace_validator.VNodeValidator,
# #                 trace_validator.MultipleCrosscutValidator,
# #                 trace_validator.UnderlappingSnapValidator,
# #                 trace_validator.TargetAreaSnapValidator,
# #             ):
# #                 print(f"Validating with {validator}")
# #                 print(f"Start time is {time.asctime()}")
# #                 if validator in [
# #                     trace_validator.GeomTypeValidator,
# #                     trace_validator.GeomNullValidator,
# #                     trace_validator.MultiJunctionValidator,
# #                 ]:
# #                     trace_gdf = validator.execute(
# #                         trace_gdf, area_gdf, auto_fix=True, parallel=True
# #                     )
# #                 else:
# #                     execute_processes.append(
# #                         pool_executor.submit(
# #                             validator.execute, *(trace_gdf, area_gdf, True, True)
# #                         )
# #                     )
# #                     execute_names.append(str(traces))
# #             error_columns = []
# #             for future, name in zip(as_completed(execute_processes), execute_names):
# #                 error_columns.append(future.result()[BaseValidator.ERROR_COLUMN])

# #         trace_gdf[BaseValidator.ERROR_COLUMN] = assemble_error_column(
# #             trace_gdf[BaseValidator.ERROR_COLUMN], error_columns
# #         )
# #         gdfs[str(traces)] = trace_gdf

# #     return gdfs


def assemble_error_column(original_column, new_columns):
    """
    Assembles error column from columns that already exist in GeoDataFrame
    and columns that are from multi-threaded execution.
    """
    assert len(original_column) == len(new_columns[0])
    mod = []
    for i in range(len(original_column)):
        orig_col = original_column[i]
        for nc in new_columns:
            orig_col += nc[i]
        mod.append(orig_col)
    assert len(mod) == len(original_column)
    return mod

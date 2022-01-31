"""
Literal wkt samples for regression testing.
"""
from ast import literal_eval
from pathlib import Path
from typing import List

from shapely.geometry import LineString
from shapely.wkt import loads


def save_to_txt(wkts: List[str], name: str) -> Path:
    """
    Save list of wkt strings to tests/sample_data/py_samples/wkts with name.txt.
    """
    assert isinstance(wkts, list)
    save_path = Path(__file__).parent / "wkts" / f"{name}.txt"
    assert wkts == literal_eval(str(wkts))
    if not save_path.exists():
        save_path.write_text(str(wkts))
    assert wkts == literal_eval(save_path.read_text())
    return save_path


def load_from_txt(name: str) -> List[str]:
    """
    Load list of wkt strings from wkts/``name``.txt.
    """
    load_path = Path(__file__).parent / "wkts" / f"{name}.txt"
    wkts = literal_eval(load_path.read_text())
    assert isinstance(wkts, list)
    assert isinstance(wkts[0], str)
    return wkts


results_in_non_simple_from_branches_and_nodes_wkt_list = load_from_txt(
    "results_in_non_simple_from_branches_and_nodes_wkt_list"
)
results_in_non_simple_from_branches_and_nodes_linestring_list = [
    loads(s) for s in results_in_non_simple_from_branches_and_nodes_wkt_list
]

mls_from_these_linestrings = load_from_txt("mls_from_these_linestrings")
mls_from_these_linestrings_list = [loads(s) for s in mls_from_these_linestrings]

results_in_sharp_turns_error_wkt = load_from_txt("results_in_sharp_turns_error_wkt")

results_in_sharp_turns_error_mls = [
    loads(wkt) for wkt in results_in_sharp_turns_error_wkt
]


results_in_multijunction_wkts = load_from_txt("results_in_multijunction_wkts")

results_in_multijunction_list_of_ls = [
    loads(ls_wkt) for ls_wkt in results_in_multijunction_wkts
]

assert all(isinstance(val, LineString) for val in results_in_multijunction_list_of_ls)


results_in_false_positive_underlapping_wkts = load_from_txt(
    "results_in_false_positive_underlapping_wkts"
)

results_in_false_positive_underlapping_ls = [
    loads(wkt) for wkt in results_in_false_positive_underlapping_wkts
]

assert all(
    [isinstance(val, LineString) for val in results_in_false_positive_underlapping_ls]
)


overlapping_mls_wkts = load_from_txt("overlapping_mls_wkts")

results_in_overlapping_ls_list = [
    list(loads(mls).geoms)[0] for mls in overlapping_mls_wkts
]


results_in_false_positive_stacked_traces_wkts = load_from_txt(
    "results_in_false_positive_stacked_traces_wkts"
)

results_in_false_positive_stacked_traces_list = [
    list(loads(mls).geoms)[0] for mls in results_in_false_positive_stacked_traces_wkts
]
# Error is caused by split causing a TypeError
# Looks like it occurs where there is an exact point for both
# linestrings at the intersection point (Y-node).
results_in_false_pos_stacked_traces = load_from_txt(
    "results_in_false_pos_stacked_traces"
)

results_in_false_pos_stacked_traces_list = [
    loads(ls) for ls in results_in_false_pos_stacked_traces
]

assert all(
    isinstance(val, LineString) for val in results_in_false_positive_stacked_traces_list
)


results_in_multijunction_why_wkts = load_from_txt("results_in_multijunction_why_wkts")

results_in_multijunction_why_ls_list = [
    loads(wkt) for wkt in results_in_multijunction_why_wkts
]


results_in_multijunction_why_wkts_2 = load_from_txt(
    "results_in_multijunction_why_wkts_2"
)

results_in_multijunction_why_ls_list_2 = [
    loads(wkt) for wkt in results_in_multijunction_why_wkts_2
]


should_result_in_target_area_underlapping_wkt = load_from_txt(
    "should_result_in_target_area_underlapping_wkt"
)


should_result_in_target_area_underlapping_wkt_poly = load_from_txt(
    "should_result_in_target_area_underlapping_wkt_poly"
)

should_result_in_target_area_underlapping_ls = loads(
    should_result_in_target_area_underlapping_wkt[0]
)

should_result_in_target_area_underlapping_poly = loads(
    should_result_in_target_area_underlapping_wkt_poly[0]
)

should_result_in_some_error_wkts = load_from_txt("should_result_in_some_error_wkts")

should_result_in_some_error_ls_list = [
    loads(wkt) for wkt in should_result_in_some_error_wkts
]

should_result_in_multij_wkts = load_from_txt("should_result_in_multij_wkts")

should_result_in_multij_ls_list = [loads(wkt) for wkt in should_result_in_multij_wkts]

should_result_in_vnode_wkts = load_from_txt("should_result_in_vnode_wkts")

should_result_in_vnode_ls_list = [loads(wkt) for wkt in should_result_in_vnode_wkts]


stacked_linestrings_wkt = load_from_txt("stacked_linestrings_wkt")

stacked_linestrings = [loads(s) for s in stacked_linestrings_wkt]


overlaps_and_cuts_self = load_from_txt("overlaps_and_cuts_self")
overlaps_and_cuts_self_linestrings = [loads(ls) for ls in overlaps_and_cuts_self]


results_in_non_simple_from_branches_and_nodes_wkt_list = load_from_txt(
    "results_in_non_simple_from_branches_and_nodes_wkt_list"
)


results_in_non_simple_from_branches_and_nodes_linestring_list = [
    loads(s) for s in results_in_non_simple_from_branches_and_nodes_wkt_list
]


very_edgy_linestring_wkt_list = load_from_txt("very_edgy_linestring_wkt_list")

very_edgy_linestring_list = [loads(s) for s in very_edgy_linestring_wkt_list]


not_so_edgy_linestrings = load_from_txt("not_so_edgy_linestrings")

not_so_edgy_linestrings_list = [loads(s) for s in not_so_edgy_linestrings]


should_or_shouldnt_pass_sharp_turns = load_from_txt(
    "should_or_shouldnt_pass_sharp_turns"
)

should_pass_sharp_turns_ls_list = [
    loads(s) for s in should_or_shouldnt_pass_sharp_turns
]


v_node_network_error = load_from_txt("v_node_network_error")

v_node_network_error_ls_list = [loads(s) for s in should_or_shouldnt_pass_sharp_turns]

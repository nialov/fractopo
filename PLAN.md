# Plan

## Changes

### 1. Empty GeoDataFrame validation (commit 8f5e812)

- Added `_check_empty_geodataframe` helper in `marimos/utils.py` that raises
  `ValueError` with a clear message when a GeoDataFrame has zero features.
- Called the helper in `parse_network_cli_args` and `parse_validation_cli_args`
  after reading traces and area files (CLI paths).
- Called the helper in `read_traces_and_area` after reading both inputs
  (interactive UI paths for both notebooks).
- Added `test_empty_traces_cli` in `tests/marimos/test_marimos.py` —
  parameterized over both notebooks, creates an empty GeoJSON, passes it via
  CLI args, and asserts the subprocess fails with `CalledProcessError`.

### 2. Filter out non-spatial layers (e.g. QGIS layer_styles)

- Added `IGNORED_LAYERS` constant (`frozenset({"layer_styles"})`) in
  `marimos/utils.py` to skip known non-spatial metadata tables.
- Added `_resolve_layer_name` helper that uses `pyogrio.list_layers` to
  enumerate layers, filters out ignored and non-geometry layers, then:
  - Returns the user-specified layer name unchanged if provided.
  - Auto-selects the layer if exactly one spatial layer exists.
  - Raises `ValueError` with available layer names if zero or multiple
    spatial layers are found.
- Called `_resolve_layer_name` in `read_spatial_app_input` before
  `gpd.read_file`, covering both notebooks' interactive paths.
- Added `import pyogrio` to `marimos/utils.py`.

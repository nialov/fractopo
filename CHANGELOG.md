# Changelog

## v0.7.0 (2024-11-1)

This release drops support for `pygeos` and Python 3.8 🐍 following
upstream packages. This has resulting in code refactoring and fixes. No
new features.

### Bug Fixes

-   **analysis:** Topological plotting fixes
-   **cli:**
    -   Fix allow_fix option for `network tracevalidate`
-   **tval:**
    -   Improve stacked (`STACKED TRACES`) detection
    -   Handle empty trace geodataframe
    -   Handle split overlap error
    -   Handle flaky `shapely` overlap detection
-   Force return types for some flaky functions (convert to `float`)
-   Remove all `pygeos` references, i.e., stop using it as it is merged
    into `shapely`

## v0.6.0 (2023-12-14)

### New Features

-   **analysis:** Allow creating contour grids without determination of
    branches and nodes i.e. topology
    ([854d1e03](https://github.com/nialov/fractopo/commit/854d1e034d932afba648bde25c41039cae6f7ccc))

### Fixes

-   **analysis:** Disable parallel processing on Windows due to
    instability
    ([614070fd](https://github.com/nialov/fractopo/commit/614070fd679e170126d40a0e66722ed92ae5853e))

## v0.5.3 (2023-05-09)

### New Features

-   A basic input geometry check is done before starting `tracevalidate` and
    `network` command-line invocations as suggested in
    <https://github.com/nialov/fractopo/issues/36>.

-   `joblib` cache settings can be set from environment
    variables.

### Fixes

-   (branches_and_nodes): Fixed `angle_to_point` by setting distance
    and similarity checks to more reasonable accuracy.

-   (general): Now crs is added before merging geodataframes in
    `dissolve_multi_part_traces`

-   (length_distribution): Handle empty arrays in in `powerlaw.Fit`
    invocations by returning `None` instead of a `Fit` instance.

-   (analysis): Length plot ticks are set explicitly in `setup_ax_for_ld` to
    avoid differences between Python package versions
    (<https://github.com/nialov/fractopo/issues/25>).

-   Z-coordinates are now *handled* across `fractopo` i.e. they do not raise
    errors. However, they are not guaranteed to be kept in results such as
    validated traces. Reported in
    <https://github.com/nialov/fractopo/issues/21>.

-   (analysis): Removed function signatures with mutable default arguments.

Full set of changes:
[`v0.5.2...v0.5.3`](https://github.com/nialov/fractopo/compare/v0.5.2...v0.5.3)

## v0.5.2 (2023-01-16)

### New Features

-   Add `plain` keyword argument to output less visualized rose and
    length plots.

### Fixes

-   Deprecated `CachedNetwork`.

-   Fixed typos in code.

### Build

-   Restructured continous integration and build structure.

    -   E.g. optimized *auxiliary* task runs on GitHub Actions and added
        a binary cache for runs.

### Docs

-   Added `./paper` directory with a manuscript describing `fractopo`.

Full set of changes:
[`v0.5.1...v0.5.2`](https://github.com/nialov/fractopo/compare/v0.5.1...v0.5.2)

## v0.5.1 (2022-11-28)

### Fixes

-   (analysis): Fixed XYI and branch ternary plot sample count to only report the
    valid node and branch counts.

-   (analysis): Fixed the cut-off vertical line to correspond to the
    actual cut-off rather than the lowest remaining length.

-   (general): Allow MultiLineStrings in `determine_valid_intersection_points`.

-   (analysis): Made sure the order of lengths used in plotting the fitted y values
    is correct in multi-scale length plots.

Full set of changes:
[`v0.5.0...v0.5.1`](https://github.com/nialov/fractopo/compare/v0.5.0...v0.5.1)

## v0.5.0 (2022-11-04)

### New Features

-   Implemented caching and parallel execution of expensive functions
    in `Network` analysis. This should speed up (repeated) runs on the
    same datasets. This implementation is based on `joblib` which provided
    the functionality without issue.

-   (analysis): Implemented plotting length data with the Probability Density
    Function (PDF) on the y-axis instead of Complementary Cumulative Number (CCM)

### Fixes

-   (analysis): Reported `CachedNetwork` deprecation

Full set of changes:
[`v0.4.1...v0.5.0`](https://github.com/nialov/fractopo/compare/v0.4.1...v0.5.0)

## v0.4.1 (2022-10-24)

### New Features

-   (cli): Save additional json data to output directory
    when running `fractopo network` for possible post-processing
    needs.

### Fixes

-   (network): General fixes to `fractopo network` command-line
    entrypoint.

-   Sort keys in any json output.

-   (network): Allow `MultiLineString` geometries when possible.
    Topologically non-valid data should be analyzeable with the
    best effort possible.

-   (branches_and_nodes): Fixes small logging bug.

Full set of changes:
[`v0.4.0...v0.4.1`](https://github.com/nialov/fractopo/compare/v0.4.0...v0.4.1)

## v0.4.0 (2022-06-17)

### New Features

-   (analysis.network): Added method (`export_network_analysis`) for
    exporting a selected set of `Network` analysis results.

-   (analysis): Added parameters for representing the real counts of traces and
    branches.

-   (general): Added `azimuth_to_unit_vector` function.

-   (analysis): Implemented rose plot functionality for non-axial data.

-   (analysis.multi_network): Added a ``MultiNetwork`` description function
    (`basic_network_descriptions_df`).

-   (analysis.multi_network): Implemented a rough first draft of a multi-scale
    length distribution fit optimizer using `scipy`.

### Fixes

-   (analysis): handle empty dataframe

-   update single dist plot

-   (analysis): change zorder

-   (analysis): handle empty array

-   (analysis): finalize multi-scale shadows

-   (analysis): show truncated length data

-   (analysis): add shadows to ternary plot points

-   (general): use latex format for units

-   (analysis): extend rose plot

-   (analysis): return polyfits from set-wise dists

-   (analysis): improve length distributions plots

-   (analysis): improved length distribution plots

-   (analysis): specify if lengths are normed

-   (analysis): pass using_branches to LineData

-   (analysis): fix basic_network_descriptions_df

-   (analysis): fix rename

-   (analysis): report min and max lengths

-   (analysis): handle extra args

-   (analysis): finalize implement of optimization

Full set of changes:
[`v0.3.0...v0.4.0`](https://github.com/nialov/fractopo/compare/v0.3.0...v0.4.0)

## v0.3.0 (2022-02-11)

### New Features

-   (analysis.multi_network): Added multi-scale azimuth set length distribution
    plotting.

-   (analysis.network): Implemented azimuth set length distribution
    plotting.

-   (analysis): Implemented a naive implementation of `Network` caching
    with `CachedNetwork` class. Errors will be raised if caching fails. User is
    recommended to fallback to `Network` in that case.

### Fixes

-   (cli): Validation column in validated trace dataset
    is now set as a `tuple` instead of as `list` to avoid conflicts
    with string representations of the data in e.g. `GeoJSON`.

Full set of changes:
[`v0.2.6...v0.3.0`](https://github.com/nialov/fractopo/compare/v0.2.6...v0.3.0)

## v0.2.6 (2022-02-03)

### New Features

-   (analysis.multi_network): Enable plotting multiple networks into the same ternary
    XYI or branch type plot.

### Fixes

-   (analysis.network): Add missing property decorator to
    `branch_lengths_powerlaw_fit_description`.

Full set of changes:
[`v0.2.5...v0.2.6`](https://github.com/nialov/fractopo/compare/v0.2.5...v0.2.6)

## v0.2.5 (2022-01-23)

Full set of changes:
[`v0.2.4...v0.2.5`](https://github.com/nialov/fractopo/compare/v0.2.4...v0.2.5)

## v0.2.4 (2022-01-13)

### New Features

-   (random_sampling): allow not determining topo

### Fixes

-   (tval): handle TypeError from split

-   (line_data): refrain from using line_gdf

-   (noxfile): setup sphinx-autobuild session

Full set of changes:
[`v0.2.3...v0.2.4`](https://github.com/nialov/fractopo/compare/v0.2.3...v0.2.4)

## v0.2.3 (2021-12-04)

### Fixes

-   (parameters): fix pie plot function

-   refactor deprecated shapely features

-   (general): check type

-   (random_sampling): get\_ methods are deprecated

-   (branches_and_nodes): set crs for outputs

Full set of changes:
[`v0.2.2...v0.2.3`](https://github.com/nialov/fractopo/compare/v0.2.2...v0.2.3)

## v0.2.2 (2021-11-12)

### New Features

-   (multi_network): add multi-scale length fit

-   (analysis): implement multiple fitters

### Fixes

-   (cli): finalize nialog implement

-   change name to non-conflicting key

-   (cli): setup logging with nialog

-   (tval): fix trace validation slowdown logging

-   (general): use static minimum line length

-   (analysis): remove usage of cached_property

-   (tval): catch TypeError from shapely split

Full set of changes:
[`v0.2.1...v0.2.2`](https://github.com/nialov/fractopo/compare/v0.2.1...v0.2.2)

## v0.2.1 (2021-09-22)

### Fixes

-   (cli): fix fractopo network entrypoint

Full set of changes:
[`v0.2.0...v0.2.1`](https://github.com/nialov/fractopo/compare/v0.2.0...v0.2.1)

## v0.2.0 (2021-09-20)

### New Features

-   (analysis): implement multiscale fit

-   implement network-cli

### Fixes

-   (analysis): handle mapped radii values

-   (cli): finish network-cli implement for now

-   deprecate safer_unary_union

-   fix truncate and circular input logic

Full set of changes:
[`v0.1.4...v0.2.0`](https://github.com/nialov/fractopo/compare/v0.1.4...v0.2.0)

## v0.1.4 (2021-08-24)

Full set of changes:
[`v0.1.3...v0.1.4`](https://github.com/nialov/fractopo/compare/v0.1.3...v0.1.4)

## v0.1.3 (2021-08-24)

### New Features

-   add network cli entrypoint

### Fixes

-   handle multipolygon geometries in efficient_clip

-   ignore geos incompatibility error

Full set of changes:
[`v0.1.2...v0.1.3`](https://github.com/nialov/fractopo/compare/v0.1.2...v0.1.3)

## v0.1.2 (2021-07-30)

Full set of changes:
[`v0.1.1...v0.1.2`](https://github.com/nialov/fractopo/compare/v0.1.1...v0.1.2)

## v0.1.1 (2021-07-29)

Full set of changes:
[`v0.1.0...v0.1.1`](https://github.com/nialov/fractopo/compare/v0.1.0...v0.1.1)

## v0.1.0 (2021-07-29)

### New Features

-   add heatmap plotting and refactor xyi plotting

-   implement additional random sampling functionality

-   (analysis): implement multinetwork class and parallel subsampling

-   all keys are np.nan by default in numerical network description

-   add plot_contour to network

### Fixes

-   minor fixes to mypy found issues

-   better parameter plotting

-   handle nan inputs in vector funcs

-   handle non-dict return

-   add verbose=False flag to powerlaw.Fit

-   specify is_filtered

-   add cli entrypoint tracevalidate

-   mauldon determination now only for circular

-   set random seed for all processes

-   specify circular target area and truncate

-   return description not full network

-   default value for no occurrences is 0, not np.nan

-   correct name for network

-   fix imports and update version

-   minor docs, style and typing fixes

-   same default snap threshold in Network as in validation

-   fix numerous pylint pointed errors

-   comment out general func for now

-   minor performance improvement and fixes

-   filter features with spatial index before gpd.clip

-   remove duplicate line

-   fix contour gridding and handle mauldon instability

-   determine branches and nodes for each cell

-   handle empty node case

### Performance improvements

-   speed up clipping with pygeos clip implement

-   improve test performace

Full set of changes:
[`v0.0.1...v0.1.0`](https://github.com/nialov/fractopo/compare/v0.0.1...v0.1.0)

## v0.0.1 (2020-12-17)

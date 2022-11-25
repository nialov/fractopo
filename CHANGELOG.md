# Changelog

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

-   improve test performance

Full set of changes:
[`v0.0.1...v0.1.0`](https://github.com/nialov/fractopo/compare/v0.0.1...v0.1.0)

## v0.0.1 (2020-12-17)

# Changelog

## v0.2.4 (2022-01-13)

Contains minor fixes and a feature update to `NetworkRandomSampler`.

#### New Features

-   (random_sampling): Allow not determining topology
    when performing random sampling using `NetworkRandomSampler`.

#### Fixes

-   (tval): Handle `TypeError` from `shapely.ops.split` in
    `split_to_determine_triangle_errors` by checking for the intersection
    between traces.

-   (line_data): Refrain from using line_gdf in operations
    as it might not contain up-to-date columns. Use `line_data`
    attributes instead.

-   (noxfile): Setup sphinx-autobuild session for automatic
    creation of documentations as source files change.

Full set of changes:
[`v0.2.3...v0.2.4`](https://github.com/nialov/fractopo/compare/v0.2.3...v0.2.4)

## v0.2.3 (2021-12-04)

#### Fixes

-   (parameters): fix pie plot function

-   refactor deprecated shapely features

-   (general): check type

-   (random_sampling): get\_ methods are deprecated

-   (branches_and_nodes): set crs for outputs

Full set of changes:
[`v0.2.2...v0.2.3`](https://github.com/nialov/fractopo/compare/v0.2.2...v0.2.3)

## v0.2.2 (2021-11-12)

#### New Features

-   (multi_network): add multi-scale length fit

-   (analysis): implement multiple fitters

#### Fixes

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

#### Fixes

-   (cli): fix fractopo network entrypoint

Full set of changes:
[`v0.2.0...v0.2.1`](https://github.com/nialov/fractopo/compare/v0.2.0...v0.2.1)

## v0.2.0 (2021-09-20)

#### New Features

-   (analysis): implement multiscale fit

-   implement network-cli

#### Fixes

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

#### New Features

-   add network cli entrypoint

#### Fixes

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

#### New Features

-   add heatmap plotting and refactor xyi plotting

-   implement additional random sampling functionality

-   (analysis): implement multinetwork class and parallel subsampling

-   all keys are np.nan by default in numerical network description

-   add plot_contour to network

#### Fixes

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

#### Performance improvements

-   speed up clipping with pygeos clip implement

-   improve test performace

Full set of changes:
[`v0.0.1...v0.1.0`](https://github.com/nialov/fractopo/compare/v0.0.1...v0.1.0)

## v0.0.1 (2020-12-17)

<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!--
Changelog titles are:
- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.
-->

# Changelog

All notable changes of this project will be documented here.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0)
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# Version 1.1.1 (September 2023)

## Fixed

- `OTGaussianProcessRegressor.predict_std` no longer returns the variance of
  the output but its standard deviation.

# Version 1.1.0 (June 2023)

## Added

- An argument `trend_type` of type `OTGaussianProcessRegressor.TREND_TYPE` to
  `OTGaussianProcessRegressor`;
  the trend type of the Gaussian process regressor can be either constant,
  linear or quadratic.
- A new optimization library `SurrogateBasedOptimization` to perform EGO-like
  surrogate-based optimization on unconstrained problems.

## Fixed

- The output of an `MLDataAcquisitionCriterion` based on a regressor built from
  constant output values is no longer `nan`.

# Version 1.0.1 (February 2022)

## Fixed

- `MLRegressorDistribution` can now use a regression algorithm instantiated with
  transformers.

# Version 1.0.0 (July 2022)

First release.

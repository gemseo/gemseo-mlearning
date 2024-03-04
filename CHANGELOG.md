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

# Develop

## Added

- [SurrogateBasedOptimization][gemseo_mlearning.algos.opt.lib_surrogate_based.SurrogateBasedOptimization]
  can use an existing [MLRegressionAlgo][gemseo.mlearning.regression.regression.MLRegressionAlgo]
  and save the [MLRegressionAlgo][gemseo.mlearning.regression.regression.MLRegressionAlgo] that it enriches
  using the `regression_file_path` argument.
- The `multi_start_n_samples`, `multi_start_algo_name` and `multi_start_algo_options` arguments of
  [OTGaussianProcessRegressor][gemseo_mlearning.regression.ot_gpr.OTGaussianProcessRegressor]
  allow to use multi-start optimization for the covariance model parameters.
- The `optimization_space` argument of
  [OTGaussianProcessRegressor][gemseo_mlearning.regression.ot_gpr.OTGaussianProcessRegressor]
  allows to set the lower and upper bounds of the covariance model parameters
  by means of a [DesignSpace][gemseo.algos.design_space.DesignSpace].
- The `covariance_model` argument of
  [OTGaussianProcessRegressor][gemseo_mlearning.regression.ot_gpr.OTGaussianProcessRegressor]
  allows to use any covariance model proposed by OpenTURNS.

## Changed

- BREAKING CHANGE: each acquisition criterion class has a specific module
  in [gemseo_mlearning.active_learning.acquisition_criteria][gemseo_mlearning.active_learning.acquisition_criteria]
  whose name is the snake-case version of it class name, i.e. `nice_criterion.py` contains `NiceCriterion`.
- BREAKING CHANGE: `gemseo_mlearning.adaptive.distribution.MLRegressorDistribution` renamed to
  [gemseo_mlearning.active_learning.distributions.base_regressor_distribution.BaseRegressorDistribution][gemseo_mlearning.active_learning.distributions.base_regressor_distribution.BaseRegressorDistribution].
- BREAKING CHANGE: `gemseo_mlearning.algos.opt.lib_surrogate_based` renamed to
  [gemseo_mlearning.algos.opt.surrogate_based_optimization][gemseo_mlearning.algos.opt.surrogate_based_optimization].
- BREAKING CHANGE: `gemseo_mlearning.algos.opt.core.surrogate_based` renamed to
  [gemseo_mlearning.algos.opt.core.surrogate_based_optimizer][gemseo_mlearning.algos.opt.core.surrogate_based_optimizer].
- BREAKING CHANGE: `MLDataAcquisition` renamed to
  [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo].
- BREAKING CHANGE: `MLDataAcquisitionCriterion` renamed to
  [BaseAcquisitionCriterion][gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion.BaseAcquisitionCriterion]
  and moved to
- [gemseo_mlearning.active_learning.acquisition_criteria][gemseo_mlearning.active_learning.acquisition_criteria].
- BREAKING CHANGE: `MLDataAcquisitionCriterionFactory` renamed to
  [AcquisitionCriterionFactory][gemseo_mlearning.active_learning.acquisition_criteria.acquisition_criterion_factory.AcquisitionCriterionFactory],
  moved to
  [gemseo_mlearning.active_learning.acquisition_criteria][gemseo_mlearning.active_learning.acquisition_criteria]
  and without the property `available_criteria` (use `AcquisitionCriterionFactory.class_names`).
- BREAKING CHANGE: `gemseo.adaptive` renamed to [gemseo_mlearning.active_learning][gemseo_mlearning.active_learning].
- BREAKING CHANGE: `gemseo.adaptive.criteria` renamed to
  [gemseo_mlearning.active_learning.acquisition_criteria][gemseo_mlearning.active_learning.acquisition_criteria].
- BREAKING CHANGE:
  The module [gemseo_mlearning.api][gemseo_mlearning.api] no longer exists;
  the functions
  [sample_discipline][gemseo_mlearning.sample_discipline]
  and [sample_disciplines][gemseo_mlearning.sample_disciplines]
  must be imported from [gemseo_mlearning][gemseo_mlearning].

# Version 1.1.2 (December 2023)

## Added

- Support for Python 3.11.
- [OTGaussianProcessRegressor][gemseo_mlearning.regression.ot_gpr.OTGaussianProcessRegressor]
  has a new optional argument `optimizer`
  to select the OpenTURNS optimizer for the covariance model parameters.

## Removed

- Support for Python 3.8.

# Version 1.1.1 (September 2023)

## Fixed

- [OTGaussianProcessRegressor.predict_std][gemseo_mlearning.regression.ot_gpr.OTGaussianProcessRegressor.predict_std]
  no longer returns the variance of the output but its standard deviation.

# Version 1.1.0 (June 2023)

## Added

- An argument `trend_type` of type
  [OTGaussianProcessRegressor.TrendType][gemseo_mlearning.regression.ot_gpr.OTGaussianProcessRegressor.TrendType]
  to [OTGaussianProcessRegressor][gemseo_mlearning.regression.ot_gpr.OTGaussianProcessRegressor];
  the trend type of the Gaussian process regressor can be either constant,
  linear or quadratic.
- A new optimization library
  [SurrogateBasedOptimization][gemseo_mlearning.algos.opt.lib_surrogate_based.SurrogateBasedOptimization]
  to perform EGO-like surrogate-based optimization on unconstrained problems.

## Fixed

- The output of an [MLDataAcquisitionCriterion][gemseo_mlearning.adaptive.criterion.MLDataAcquisitionCriterion]
  based on a regressor built from constant output values is no longer `nan`.

# Version 1.0.1 (February 2022)

## Fixed

- [BaseRegressorDistribution][gemseo_mlearning.adaptive.distribution.BaseRegressorDistribution]
  can now use a regression algorithm instantiated with transformers.

# Version 1.0.0 (July 2022)

First release.

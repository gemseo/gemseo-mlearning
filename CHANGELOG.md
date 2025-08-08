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

## Version 3.0.0 (August 2025)

### Added

- [gemseo_mlearning.settings.opt][gemseo_mlearning.settings.opt]: the Pydantic models to define the settings of the optimization algorithms.
- [gemseo_mlearning.settings.mlearning][gemseo_mlearning.settings.mlearning]: the Pydantic models to define the settings of the machine learning algorithms.

### Changed

- `SBOSettings` renamed to `SBO_Settings`.
- `SMTEGOSettings` renamed to `SMT_EGO_Settings`.
- `SMTRegressorSettings` renamed to `SMT_Regressor_Settings`.

## Version 2.0.1 (April 2025)

### Fixed

- The optimization algorithm `SMT_EGO` correctly uses the option ``normalize_design_space``.
- The method
  [ActiveLearningAlgo.acquire_new_points][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo.acquire_new_points]
  works when the discipline it uses has output variables
  that are not outputs of the regression model
  used by [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo]

## Version 2.0.0 (November 2024)

### Added

- Support GEMSEO v6.
- Support for Python 3.12.
- [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo]
  can acquire points by batch using the `batch_size`
  and `mc_size` argument,
  when the regressor is based on a random process
  such as `GaussianProcessRegressor`
  and `OTGaussianProcessRegressor`.
  This option is only available for criteria dedicated
  to level set [LevelSet][gemseo_mlearning.active_learning.acquisition_criteria.level_set.level_set.LevelSet]
  (alternatively quantile estimation [Quantile][gemseo_mlearning.active_learning.acquisition_criteria.quantile.quantile.Quantile])
  and the expected improvement for maximum/minimum estimation [Maximum][gemseo_mlearning.active_learning.acquisition_criteria.maximum].
- The [Branin][gemseo_mlearning.problems.branin] and
  [Rosenbrock][gemseo_mlearning.problems.rosenbrock] problems
  can be used to benchmark
  the efficiency of the active learning algorithms
  and estimate several quantities of interest,
  optimas or quantiles for instance.
- [AcquisitionView][gemseo_mlearning.active_learning.visualization.acquisition_view.AcquisitionView]
  can be used to plot
  both the output of the original model,
  the prediction of the surrogate model,
  the standard deviation of the surrogate model
  and the acquisition criterion,
  when the input dimension is 1 or 2.
- [SMTRegressor][gemseo_mlearning.regression.smt_regressor.SMTRegressor]
  can be any surrogate model available in the Python package [SMT](https://smt.readthedocs.io/).
- `"SMT_EGO"` is the name of the expected global optimization (EGO) algorithm
  wrapping the surrogate-based optimizers available in the Python package [SMT](https://smt.readthedocs.io/);
  this is the unique algorithm of the [SMTEGO][gemseo_mlearning.algos.opt.smt.smt_ego.SMTEGO] optimization library.
- [SurrogateBasedOptimization][gemseo_mlearning.algos.opt.surrogate_based_optimization.SurrogateBasedOptimization]
  can use the acquisition criteria `CB` and `Mean` in addition to `EI`.
- [SurrogateBasedOptimization][gemseo_mlearning.algos.opt.surrogate_based_optimization.SurrogateBasedOptimization]
  can use an existing [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor]
  and save the [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor] that it enriches
  using the `regression_file_path` option.

### Changed

- BREAKING CHANGE: The acquisition algorithm settings have to be passed to
  [SurrogateBasedOptimizer][gemseo_mlearning.algos.opt.core.surrogate_based_optimizer.SurrogateBasedOptimizer]
  as keyword arguments, *i.e.* `SurrogateBasedOptimizer(..., key_1=value_1, key_2=value_2, ...)`.
- BREAKING CHANGE: The term `option` has been replaced by `setting` when it was linked to a DOE or an optimization algorithm.
- BREAKING CHANGE: The argument `distribution` of
  [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo].
  renamed to `regressor`;
  it can be either a
- [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor]
  or a
  [BaseRegressorDistribution][gemseo_mlearning.active_learning.distributions.base_regressor_distribution.BaseRegressorDistribution].
- BREAKING CHANGE: The method `compute_next_input_data` of
  [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo].
  renamed to
  [find_next_point][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo.find_next_point].
- BREAKING CHANGE: The method `update_algo` of
  [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo].
  renamed to
  [acquire_new_points][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo.acquire_new_points].
- BREAKING CHANGE: `MaxExpectedImprovement` renamed to ``Maximum`.
- BREAKING CHANGE: `MinExpectedImprovement` renamed to ``Minimum`.
- BREAKING CHANGE: `ExpectedImprovement` removed.
- BREAKING CHANGE: the acquisition criterion `LimitState` renamed to
  [LevelSet][gemseo_mlearning.active_learning.acquisition_criteria.level_set.level_set.LevelSet]
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
  [BaseAcquisitionCriterionFactory][gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion_family.BaseAcquisitionCriterionFactory],
  moved to
  [gemseo_mlearning.active_learning.acquisition_criteria][gemseo_mlearning.active_learning.acquisition_criteria]
  and without the property `available_criteria` (use `BaseAcquisitionCriterionFactory.class_names`).
- BREAKING CHANGE: `gemseo.adaptive` renamed to [gemseo_mlearning.active_learning][gemseo_mlearning.active_learning].
- BREAKING CHANGE: `gemseo.adaptive.criteria` renamed to
  [gemseo_mlearning.active_learning.acquisition_criteria][gemseo_mlearning.active_learning.acquisition_criteria].

### Fixed

- The data transformer can be set with the `"transformer"` key of the `regression_options` dictionary
  passed to [SurrogateBasedOptimization][gemseo_mlearning.algos.opt.surrogate_based_optimization.SurrogateBasedOptimization].
- The [Quantile][gemseo_mlearning.active_learning.acquisition_criteria.quantile]
  estimates the quantile by Monte Carlo sampling
  by means of the probability distributions of the input variables;
  these distributions are defined with its new argument `uncertain_space`.

### Removed

- `AcquisitionCriterionFactory`; replaced by different factories for the developers.
- `sample_discipline`; use [sample_disciplines][gemseo.sample_disciplines] from `gemseo` instead.
- `sample_disciplines`; move to `gemseo`: [sample_disciplines][gemseo.sample_disciplines].
- `MAEMeasure`; moved to `gemseo`: [MAEMeasure][gemseo.mlearning.regression.quality.mae_measure.MAEMeasure].
- `MEMeasure`; moved to `gemseo`: [MEMeasure][gemseo.mlearning.regression.quality.me_measure.MEMeasure].
- `GradientBoostingRegressor`; moved to `gemseo`: [GradientBoostingRegressor][gemseo.mlearning.regression.algos.gradient_boosting.GradientBoostingRegressor].
- `MLPRegressor`; moved to `gemseo`: [MLPRegressor][gemseo.mlearning.regression.algos.mlp.MLPRegressor].
- `OTGaussianProcessRegressor`; moved to `gemseo`: [OTGaussianProcessRegressor][gemseo.mlearning.regression.algos.ot_gpr.OTGaussianProcessRegressor].
- `RegressorChain`; moved to `gemseo`: [RegressorChain][gemseo.mlearning.regression.algos.regressor_chain.RegressorChain].
- `SVMRegressor`; moved to `gemseo`: [SVMRegressor][gemseo.mlearning.regression.algos.svm.SVMRegressor].
- `TPSRegressor`; moved to `gemseo`: [TPSRegressor][gemseo.mlearning.regression.algos.thin_plate_spline.TPSRegressor].

## Version 1.1.2 (December 2023)

### Added

- Support for Python 3.11.
- `OTGaussianProcessRegressor` has a new optional argument `optimizer`
  to select the OpenTURNS optimizer for the covariance model parameters.

### Removed

- Support for Python 3.8.

## Version 1.1.1 (September 2023)

### Fixed

- `OTGaussianProcessRegressor.predict_std`
  no longer returns the variance of the output but its standard deviation.

## Version 1.1.0 (June 2023)

### Added

- An argument `trend_type` of type `OTGaussianProcessRegressor.TrendType` to `OTGaussianProcessRegressor`;
  the trend type of the Gaussian process regressor can be either constant,
  linear or quadratic.
- A new optimization library
  [SurrogateBasedOptimization][gemseo_mlearning.algos.opt.surrogate_based_optimization.SurrogateBasedOptimization]
  to perform EGO-like surrogate-based optimization on unconstrained problems.

### Fixed

- The output of an `MLDataAcquisitionCriterion`
  based on a regressor built from constant output values is no longer `nan`.

## Version 1.0.1 (February 2022)

### Fixed

- [BaseRegressorDistribution][gemseo_mlearning.active_learning.distributions.base_regressor_distribution.BaseRegressorDistribution]
  can now use a regression algorithm instantiated with transformers.

## Version 1.0.0 (July 2022)

First release.

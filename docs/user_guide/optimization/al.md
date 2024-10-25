<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->
# Surrogate-based optimizers using the AL algorithm

The [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo]
presented in [this page](../active_learning/active_learning_algo.md)
can be used to minimize a cost function using the family of acquisition criteria
[Minimum][gemseo_mlearning.active_learning.acquisition_criteria.minimum.minimum.Minimum]

[SurrogateBasedOptimization][gemseo_mlearning.algos.opt.surrogate_based_optimization.SurrogateBasedOptimization]
is a collection of surrogate-based optimization algorithms
built on top of [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo]
and [Minimum][gemseo_mlearning.active_learning.acquisition_criteria.minimum.minimum.Minimum].

## Basic usage

[SurrogateBasedOptimization][gemseo_mlearning.algos.opt.surrogate_based_optimization.SurrogateBasedOptimization]
includes a single optimization algorithm,
called `"SBO"`.

Given a maximum number of iterations,
it can be used as is
by any [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]:
```python
execute_algo(optimization_problem, "SBO", max_iter=50)
```
and any [MDOScenario][gemseo.scenarios.mdo_scenario.MDOScenario]:
```python
scenario.execute({"algo_name": "SBO", "max_iter": 50})
```

In this case,
the settings are

- the expected improvement as acquisition criterion,
- 1 point acquired at a time,
- the [OTGaussianProcessRegressor][gemseo.mlearning.regression.algos.ot_gpr.OTGaussianProcessRegressor]
  wrapping the Kriging model from OpenTURNS,
- 10 initial training points based on an optimized latin hypercube sampling (LHS) technique,
- a multi-start local optimization of the acquisition criterion
  with a gradient-based algorithm
  from 20 start points with a limit of 200 iterations per local optimization.

## Settings

In the following,
the training output values already acquired
and the output and uncertainty predictions at a given input point $x$
are respectively denoted $\{y_1,\ldots,y_n\}$,
$\mu(x)$ and $\sigma(x)$.

### General

::: gemseo_mlearning.algos.opt.sbo_settings.SBOSettings
    options:
      show_root_heading: false
      show_bases: false
      show_root_toc_entry: false

### Acquisition criteria

The three acquisition criteria are

| Value      | Name                   | Expression                                     |
|------------|------------------------|------------------------------------------------|
| `"EI"`     | Expected improvement   | $\mathbb{E}[\max(\min(y_1,\dots,y_n)-Y(x),0]$     |
| `"LCB"`    | Lower confidence bound | $\mu(x)-\kappa\times\sigma(x)$ with $\kappa>0$ |
| `"Output"` | Mean output            | $\mu(x)$                                       |

where $Y$ is a random process modelling the uncertainty of the surrogate model $\hat{f}$ and
$\mu(x)\equiv\mathbb{E}[Y(x)]$ and $\sigma(x)\equiv\mathbb{S}[Y(x)]$ denote
the expectation and the standard deviation of $Y(x)$ at input point $x$.
Most of the time,
$Y$ is a Gaussian process and $\hat{f}$ is a Kriging model.

### Initial training dataset

Any DOE algorithm known by GEMSEO can be used to generate the initial training dataset.

### Parallel acquisition

Points can be acquired by batch of $q>1$ points,
as Kriging is well-suited to parallelize optimization[@ginsbourger2010kriging].
To this aim,
when `batch_size` is greater than 1,
[ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo]
uses a parallel version of the expected improvement criterion.
Unfortunately,
the latter has no analytic expression
and the acquisition criterion is thus instead evaluated with crude Monte-Carlo.

### Surrogate models

[SurrogateBasedOptimization][gemseo_mlearning.algos.opt.surrogate_based_optimization.SurrogateBasedOptimization]
is compatible with all regressors,
whose classes derive from [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor].

It can also be used from an existing surrogate models.
In this case, it will skip the construction and learning of the initial training dataset.

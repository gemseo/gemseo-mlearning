<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->
# Active learning

## API

The [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo] class
defines an active learning algorithm.

Given a coarse regressor $\hat{f}$ (a.k.a. surrogate model)
of a model $f$ (a.k.a. substituted model),
this algorithm updates this regressor sequentially
from input-output points maximizing (or minimizing) an acquisition criterion
(a.k.a. infill criterion).

There are five families of acquisition criteria:

- [Minimum][gemseo_mlearning.active_learning.acquisition_criteria.minimum.minimum.Minimum]:
  make the minimum of the surrogate model
  tends towards the minimum of the substituted model
  when the number of acquired points tends to infinity,
- [Maximum][gemseo_mlearning.active_learning.acquisition_criteria.maximum.maximum.Maximum]:
  make the maximum of the surrogate model
  tends towards the maximum of the substituted model
  when the number of acquired points tends to infinity,
- [LevelSet][gemseo_mlearning.active_learning.acquisition_criteria.level_set.level_set.LevelSet]:
  make a level set of the surrogate model
  tends towards the corresponding level set of the substituted model
  when the number of acquired points tends to infinity,
- [Quantile][gemseo_mlearning.active_learning.acquisition_criteria.quantile.quantile.Quantile]:
  make a quantile of the surrogate model
  tends towards the quantile of the substituted model
  when the number of acquired points tends to infinity,
- [Exploration][gemseo_mlearning.active_learning.acquisition_criteria.exploration.exploration.Exploration]:
  make the error of the surrogate model tends towards zero
  when the number of acquired points tends to infinity.

Many of these acquisition criteria consider that
the surrogate model is a Gaussian process (GP) regressor $\hat{f}$
and are expressed from realizations or statistics of this GP $\hat{F}$,
*e.g.* its mean $\hat{f}(x)$ and its variance $\hat{k}(x,x)$ at $x$
where $\hat{k}$ is its covariance function.
These statistics and realizations can be obtained using the
[KrigingDistribution][gemseo_mlearning.active_learning.distributions.kriging_distribution.KrigingDistribution]
class
which can be built from any regressor deriving from
[BaseRandomProcessRegressor][gemseo.mlearning.regression.algos.base_random_process_regressor.BaseRandomProcessRegressor].
By *distribution* we mean
the probability distribution of a random function of which $f$ is an instance.
For non-GP regressors,
this distribution is not a random process from the literature
but an empirical distribution based on resampling techniques
and is qualified as *universal* by its authors:
[RegressorDistribution][gemseo_mlearning.active_learning.distributions.regressor_distribution.RegressorDistribution].

A basic use of this class is

1. instantiate
   the [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo]
   from an input space of type [DesignSpace][gemseo.algos.design_space.DesignSpace],
   a [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor]
   and the name of a family of acquisition criteria
   (a default acquisition criterion will be set accordingly),
2. update the regressor with the method ``acquire_new_points``,
3. get the updated regressor with the attribute ``regressor``.

For more advanced use,
it is possible to change the acquisition algorithm,
i.e. the optimization algorithm to minimize or maximize the acquisition criterion,
as well as the acquisition criterion among the selected family.

Lastly,
the [visualization][gemseo_mlearning.active_learning.visualization] subpackage
offers plotting capabilities
to draw the evolution of both the surrogate model and the acquisition criterion.

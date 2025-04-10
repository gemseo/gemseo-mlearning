<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Active learning algorithm

The [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo] class
defines an [active learning](what_active_learning_is.md) algorithm.

## In a nutshell

Given a coarse regressor $\hat{f}$ (a.k.a. surrogate model)
of a model $f$ (a.k.a. substituted model),
this algorithm updates this regressor sequentially
from input-output points maximizing (or minimizing) an acquisition criterion
(a.k.a. infill criterion)
chosen for a specific purpose.

## Choosing an acquisition criterion

`gemseo-mlearning` includes five families of acquisition criteria corresponding to as many purposes:

- the [Minimum][gemseo_mlearning.active_learning.acquisition_criteria.minimum.minimum.Minimum] family
  aims to make the minimum of the surrogate model
  tends towards the minimum of the substituted model
  when the number of acquired points tends to infinity,
- the [Maximum][gemseo_mlearning.active_learning.acquisition_criteria.maximum.maximum.Maximum] family
  aims to make the maximum of the surrogate model
  tends towards the maximum of the substituted model
  when the number of acquired points tends to infinity,
- the [LevelSet][gemseo_mlearning.active_learning.acquisition_criteria.level_set.level_set.LevelSet] family
  aims to make a level set of the surrogate model
  tends towards the corresponding level set of the substituted model
  when the number of acquired points tends to infinity,
- the [Quantile][gemseo_mlearning.active_learning.acquisition_criteria.quantile.quantile.Quantile] family
  aims to make a quantile of the surrogate model
  tends towards the corresponding quantile of the substituted model
  when the number of acquired points tends to infinity,
- the [Exploration][gemseo_mlearning.active_learning.acquisition_criteria.exploration.exploration.Exploration] family
  aims to make the error of the surrogate model tends towards zero
  when the number of acquired points tends to infinity.

[Minimum][gemseo_mlearning.active_learning.acquisition_criteria.minimum.minimum.Minimum],
[Maximum][gemseo_mlearning.active_learning.acquisition_criteria.maximum.maximum.Maximum]
and [Quantile][gemseo_mlearning.active_learning.acquisition_criteria.quantile.quantile.Quantile]
provides an estimation of the quantity of interest,
namely the minimum, the maximum and a quantile respectively.
This value is updated each time a training point is acquired
and can be accessed via the attribute
[qoi][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo.qoi].

Whatever the family,
the surrogate model can be used for prediction as any surrogate model.
However,
outside the
[Exploration][gemseo_mlearning.active_learning.acquisition_criteria.exploration.exploration.Exploration] family,
it is important to bear in mind
that its learning dataset has been designed to estimate a particular quantity of interest,
and should therefore be used with caution.
In other words,
a surrogate model built to find the minimum can be bad at predicting the high output values.
In such cases,
the predicted standard-deviation
that characterizes the uncertainty of the prediction
can help to judge the accuracy of the surrogate prediction.

## Choosing a surrogate model

Many of these acquisition criteria consider that
the surrogate model is a Gaussian process (GP) regressor $\hat{f}$
and are expressed from realizations or statistics of this GP $\hat{F}$,
*e.g.* its mean $m_n(x)$ and its variance $c_n(x,x)$ at $x$
where $c_n(\cdot,\cdot)$ is its covariance function.
These statistics and realizations can be obtained using the
[KrigingDistribution][gemseo_mlearning.active_learning.distributions.kriging_distribution.KrigingDistribution]
class
which can be built from any regressor deriving from
[BaseRandomProcessRegressor][gemseo.mlearning.regression.algos.base_random_process_regressor.BaseRandomProcessRegressor],
such as [GaussianProcessRegressor][gemseo.mlearning.regression.algos.gpr.GaussianProcessRegressor] based on
[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)
and [OTGaussianProcessRegressor][gemseo.mlearning.regression.algos.ot_gpr.OTGaussianProcessRegressor] based on
[OpenTURNS](https://openturns.github.io/openturns/latest/user_manual/response_surface/_generated/openturns.KrigingAlgorithm.html).
By *distribution* we mean
the probability distribution of a random function of which $f$ is an instance.
For non-GP regressors,
this distribution is not a random process from the literature
but an empirical distribution based on resampling techniques
and is qualified as *universal* by its authors:
[RegressorDistribution][gemseo_mlearning.active_learning.distributions.regressor_distribution.RegressorDistribution].

## How to use this algorithm

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
They can be easily accessed
from the [ActiveLearningAlgo][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo]
using its methods:

- [plot_qoi_history][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo.plot_qoi_history]
  to visualize the estimation of the quantity of interest in function of the acquisition step,
- [plot_acquisition_view][gemseo_mlearning.active_learning.active_learning_algo.ActiveLearningAlgo.plot_acquisition_view]
  to visualize the discipline, the regressor, the acquisition criterion and the standard deviation
  in the case of two scalar input variables.

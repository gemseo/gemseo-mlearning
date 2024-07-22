# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
r"""Active learning.

The main class defines an active learning algorithm:
[ActiveLearningAlgo][gemseo_mlearning.active_learning_algo.ActiveLearningAlgo].

Given a coarse regressor $\hat{f}$ (a.k.a. surrogate model)
of a model $f$ (a.k.a. substituted model),
this algorithm updates this regressor sequentially
from input-output points maximizing (or minimizing) an acquisition criterion
(a.k.a. infill criterion).

There are five families of acquisition criteria:

- [Minimum][gemseo_mlearning.active_learning_algo.acquisition_criteria.minimum.Minimum]:
  make the minimum of the surrogate model
  tends towards the minimum of the substituted model
  when the number of acquired points tends to infinity,
- [Maximum][gemseo_mlearning.active_learning_algo.acquisition_criteria.maximum.Maximum]:
  make the maximum of the surrogate model
  tends towards the maximum of the substituted model
  when the number of acquired points tends to infinity,
- [LevelSet]
  [gemseo_mlearning.active_learning_algo.acquisition_criteria.level_set.LevelSet]:
  make a level set of the surrogate model
  tends towards the corresponding level set of the substituted model
  when the number of acquired points tends to infinity,
- [Quantile]
  [gemseo_mlearning.active_learning_algo.acquisition_criteria.quantile.Quantile]:
  make a quantile of the surrogate model
  tends towards the quantile of the substituted model
  when the number of acquired points tends to infinity,
- [Exploration]
  [gemseo_mlearning.active_learning_algo.acquisition_criteria.exploration.Exploration]:
  make the error of the surrogate model tends towards zero
  when the number of acquired points tends to infinity.

Many of these acquisition criteria consider that
the surrogate model is a Gaussian process (GP) regressor $\hat{f}$
and are expressed from realizations or statistics of this GP $\hat{F}$,
e.g. its mean $\hat{f}(x)$ and its variance $\hat{k}(x,x)$ at $x$
where $\hat{k}$ is its covariance function.
These statistics and realizations can be obtained using the
[KrigingDistribution][gemseo_mlearning.active_learning.distribution.kriging_distribution.KrigingDistribution]
class
which can be built from any regressor deriving from
[BaseRandomProcessRegressor][gemseo.mlearning.regression.algos.base_random_process_regressor.BaseRandomProcessRegressor].
By *distribution* we mean
the probability distribution of a random function of which $f$ is an instance.
For non-GP regressors,
this distribution is not a random process from the literature
but an empirical distribution based on resampling techniques
and is qualified as *universal* by its authors:
[RegressorDistribution][gemseo_mlearning.active_learning.distribution.regressor_distribution.RegressorDistribution].

A basic use of this class is

1. instantiate
   the [ActiveLearningAlgo][gemseo_mlearning.active_learning_algo.ActiveLearningAlgo]
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
"""

from __future__ import annotations

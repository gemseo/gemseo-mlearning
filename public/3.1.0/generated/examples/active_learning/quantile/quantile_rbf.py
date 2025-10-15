# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""# Non-GP regressor."""

from __future__ import annotations

from gemseo import configuration
from gemseo import sample_disciplines
from gemseo.mlearning.regression.algos.rbf import RBFRegressor
from gemseo.uncertainty.statistics.empirical_statistics import EmpiricalStatistics

from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo
from gemseo_mlearning.active_learning.distributions.regressor_distribution import (
    RegressorDistribution,
)
from gemseo_mlearning.problems.rosenbrock.rosenbrock_discipline import (
    RosenbrockDiscipline,
)
from gemseo_mlearning.problems.rosenbrock.rosenbrock_space import RosenbrockSpace

# Update the configuration of |g| to speed up the script.
configuration.fast = True

# %%
# The use of active learning methods
# dedicated to quantile estimation
# is illustrated in this example,
# with all default settings.
# The function with the quantile of interest is
# the Rosenbrock function $f(x_1,x_2)=(1-x_1)^2+100(x_2-x_1^2)^2$:
discipline = RosenbrockDiscipline()
# %%
# with $x_1$ and $x_2$ uniformly distributed over $[-2,2]^2$:
uncertain_space = RosenbrockSpace()

# %%
# First,
# we create an initial training dataset using an optimal LHS including 10 samples:
learning_dataset = sample_disciplines(
    [discipline], uncertain_space, "y", algo_name="OT_OPT_LHS", n_samples=10
)

# %%
# and a universal regressor, namely a radial basis function network based on SciPy:
regressor = RBFRegressor(learning_dataset)
regressor_distribution = RegressorDistribution(regressor, bootstrap=False)
regressor_distribution.learn()

# %%
# Then,
# we look for 20 points that will help us
# to approximate the 35% quantile.
# By default,
# for this purpose,
# the active learning algorithm looks
# for the point minimizing the U-function
# with the help of the SLSQP gradient-based algorithm
# applied in a multistart framework.
level = 0.35
active_learning = ActiveLearningAlgo(
    "Quantile",
    uncertain_space,
    regressor_distribution,
    level=level,
    uncertain_space=uncertain_space,
)
active_learning.acquire_new_points(discipline, 10)


# %%
# Then,
# we plot the history of the quantity of interest
active_learning.plot_qoi_history()
# %%
# as well as
# the training points,
# the original model,
# the RBF regressor
# and the U-function
# after the last acquisition:
active_learning.plot_acquisition_view(discipline=discipline)

# %%
# Finally,
# we compare the estimated quantile
# from the active learning procedure
# to the Monte Carlo estimate
# for both algorithms
dataset = sample_disciplines(
    [discipline], uncertain_space, "y", algo_name="OT_MONTE_CARLO", n_samples=10000
)
reference_quantile = EmpiricalStatistics(dataset, ["y"]).compute_quantile(level)

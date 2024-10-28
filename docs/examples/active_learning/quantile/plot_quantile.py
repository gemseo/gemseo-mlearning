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
"""# Quantile estimation using a Gaussian process regressor"""

from __future__ import annotations

from gemseo import configure
from gemseo import configure_logger
from gemseo import sample_disciplines
from gemseo.mlearning.regression.algos.ot_gpr import OTGaussianProcessRegressor
from gemseo.uncertainty.statistics.empirical_statistics import EmpiricalStatistics

from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo
from gemseo_mlearning.problems.rosenbrock.rosenbrock_discipline import (
    RosenbrockDiscipline,
)
from gemseo_mlearning.problems.rosenbrock.rosenbrock_space import RosenbrockSpace

configure(False, False, True, False, False, False, False)
configure_logger()

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
    [discipline], uncertain_space, "y", 10, "OT_OPT_LHS"
)

# %%
# and an initial Gaussian process regressor from OpenTURNS:
regressor = OTGaussianProcessRegressor(
    learning_dataset,
    trend="quadratic",
)

# %%
# Then,
# we look for 20 points that will help us
# to approximate the 35% quantile.
# By default,
# for this purpose,
# the active learning algorithm looks
# for the point minimizing the U-function
# with the help of the SLSQP algorithm
# applied in a multistart framework.
level = 0.35
active_learning = ActiveLearningAlgo(
    "Quantile",
    uncertain_space,
    regressor,
    level=level,
    uncertain_space=uncertain_space,
)
active_learning.acquire_new_points(discipline, 20)

# %%
# Then,
# we plot the history of the quantity of interest
active_learning.plot_qoi_history(file_path="tot.png")
# %%
# as well as
# the training points,
# the original model,
# the Gaussian process regressor
# and the U-function
# after the last acquisition:
active_learning.plot_acquisition_view(discipline=discipline)

# %%
# Finally,
# we compare the estimated quantile
# from the active learning procedure
# to the Monte Carlo estimate
# for both algorithms:
dataset = sample_disciplines(
    [discipline], uncertain_space, "y", 10000, "OT_MONTE_CARLO"
)
reference_quantile = EmpiricalStatistics(dataset, ["y"]).compute_quantile(level)
print(reference_quantile, active_learning.qoi)

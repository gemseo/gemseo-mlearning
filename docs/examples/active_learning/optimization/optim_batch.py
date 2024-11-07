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
"""# Batch acquisition."""

from __future__ import annotations

import matplotlib.pyplot as plt
from gemseo import configure
from gemseo import configure_logger
from gemseo import sample_disciplines
from gemseo.mlearning.regression.algos.ot_gpr import OTGaussianProcessRegressor
from numpy import argmin
from numpy import concatenate
from numpy import unique

from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo
from gemseo_mlearning.problems.rosenbrock.rosenbrock_discipline import (
    RosenbrockDiscipline,
)
from gemseo_mlearning.problems.rosenbrock.rosenbrock_space import RosenbrockSpace

# Update the configuration of |g| to speed up the script (use configure() with care)
configure(False, False, True, False, False, False, False)

configure_logger()

# %%
# The use of active learning methods
# dedicated to optimization
# is illustrated in this example.
# More specifically,
# we aim to test here
# the impact of batch acquisition,
# that is to say when several points
# are added at each method iteration,
# on the active learning procedure.
# The function to minimize is
# the Rosenbrock function $f(x_1,x_2)=(1-x_1)^2+100(x_2-x_1^2)^2$:
discipline = RosenbrockDiscipline()
# %%
# with $x_1$ and $x_2$ belonging to $[-2,2]^2$:
input_space = RosenbrockSpace()

# %%
# First,
# we create an initial training dataset using an optimal LHS including 10 samples:
learning_dataset = sample_disciplines(
    [discipline], input_space, "y", algo_name="OT_OPT_LHS", n_samples=10
)

# %%
# and two identical initial
# Gaussian process regressors from OpenTURNS:
regressor_1 = OTGaussianProcessRegressor(
    learning_dataset, trend="quadratic", multi_start_n_samples=20
)
regressor_2 = OTGaussianProcessRegressor(
    learning_dataset, trend="quadratic", multi_start_n_samples=20
)

# %%
# Then,
# we build two active learning algorithms
# to test the impact of batch acquisition,
# that is to say when several points
# are added at each method iteration,
# on the active learning procedure.
# We consider one algorithm with
# samples added one by one
# also called sequential acquisition (default),
# and in the second one,
# four samples instead of one
# are added at each iteration (parallel acquisition),
# using argument `batch_size` set to 4.
# Note that here,
# this feature is only available
# with the expected improvement criterion.
# All other settings are put to
# their default values.
active_learning_1 = ActiveLearningAlgo("Minimum", input_space, regressor_1)
active_learning_2 = ActiveLearningAlgo(
    "Minimum", input_space, regressor_2, batch_size=4
)
active_learning_1.acquire_new_points(discipline, n_samples=20)
active_learning_2.acquire_new_points(discipline, n_samples=20)

# %%
# To study the results,
# we extract first
# the data associated
# to the history
# of the quantity of interest
# for both active learning procedures
history_1 = active_learning_1.qoi_history
history_2 = active_learning_2.qoi_history
# and we compare them in a plot
plt.plot(history_1[0], concatenate(history_1[1]), marker="o", label="Sequential")
plt.plot(history_2[0], concatenate(history_2[1]), marker="o", label="Parallel")
plt.xlabel("Number of evaluations")
plt.ylabel("Minimum")
plt.legend()
plt.show()

# %%
# We can also
# compare the estimated optimas
# from the active learning procedures
# to their exact counterparts
# for both algorithms

# %%
# Finally,
# for both active learning algorithms,
# we plot the training points,
# alongside the original model.

# Creation of the grid
# and estimation of the different quantities
n_test = 10
observations = sample_disciplines(
    [discipline], input_space, "y", algo_name="OT_FULLFACT", n_samples=n_test**2
).values

# Plotting the exact minimum and the estimated minima
# alongside the learning points
plt.figure()
points_1 = active_learning_1.regressor.learning_set.to_numpy()
points_2 = active_learning_2.regressor.learning_set.to_numpy()
plt.contour(
    unique(observations[:, 0]),
    unique(observations[:, 1]),
    observations[:, 2].reshape(n_test, n_test),
)
bar = plt.colorbar()
bar.set_label(r"$f(x_1,x_2)$")
plt.scatter([1], [1], marker="o", label="Exact minimum", color="red")
plt.scatter(
    points_1[argmin(points_1[:, -1]), 0],
    points_1[argmin(points_1[:, -1]), 1],
    marker="o",
    label="Minimum with sequential acquisition",
    color="tab:blue",
)
plt.scatter(
    points_2[argmin(points_2[:, -1]), 0],
    points_2[argmin(points_2[:, -1]), 1],
    marker="o",
    label="Minimum with parallel acquisition",
    color="tab:orange",
)
plt.scatter(
    points_1[:, 0],
    points_1[:, 1],
    marker="*",
    label="Learning points from algo with sequential acquisition",
)
plt.scatter(
    points_2[:, 0],
    points_2[:, 1],
    marker="*",
    label="Learning points from algo with parallel acquisition",
)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.legend()
plt.show()

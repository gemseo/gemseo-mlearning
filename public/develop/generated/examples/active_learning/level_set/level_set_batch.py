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
from gemseo.disciplines.surrogate import SurrogateDiscipline
from gemseo.mlearning.regression.algos.ot_gpr import OTGaussianProcessRegressor
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
# dedicated to level set estimation
# is illustrated in this example.
# More specifically,
# we aim to test here
# the impact of batch acquisition,
# that is to say when several points
# are added at each method iteration,
# on the active learning procedure.
# The function with the level set of interest is
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
regressor_1 = OTGaussianProcessRegressor(learning_dataset, trend="quadratic")
regressor_2 = OTGaussianProcessRegressor(learning_dataset, trend="quadratic")

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
# All other settings are put to
# their default values.
value_level = 400
active_learning_1 = ActiveLearningAlgo(
    "LevelSet", input_space, regressor_1, output_value=value_level
)
active_learning_2 = ActiveLearningAlgo(
    "LevelSet", input_space, regressor_2, output_value=value_level, batch_size=4
)
active_learning_1.acquire_new_points(discipline, n_samples=20)
active_learning_2.acquire_new_points(discipline, n_samples=20)

# %%
# To study the results,
# for both active learning algorithms,
# we plot the training points,
# the estimated level sets
# alongside the original model.

# Creation of the grid
# and estimation of the different quantities
n_test = 10
surrogate_1 = SurrogateDiscipline(active_learning_1.regressor)
surrogate_2 = SurrogateDiscipline(active_learning_2.regressor)
observations = sample_disciplines(
    [discipline], input_space, "y", algo_name="OT_FULLFACT", n_samples=n_test**2
).values
observations_gp_1 = sample_disciplines(
    [surrogate_1], input_space, "y", algo_name="OT_FULLFACT", n_samples=n_test**2
).values
observations_gp_2 = sample_disciplines(
    [surrogate_2], input_space, "y", algo_name="OT_FULLFACT", n_samples=n_test**2
).values

# Plotting the contours of the Rosenbrock function
# alongside the learning points
# and the level sets.
plt.figure()
points_1 = active_learning_1.regressor.learning_set.to_numpy()
points_2 = active_learning_2.regressor.learning_set.to_numpy()
level_set_exact = plt.contour(
    unique(observations[:, 0]),
    unique(observations[:, 1]),
    observations[:, 2].reshape(n_test, n_test),
    levels=[value_level],
    colors="red",
)
level_set_gp_1 = plt.contour(
    unique(observations_gp_1[:, 0]),
    unique(observations_gp_1[:, 1]),
    observations_gp_1[:, 2].reshape(n_test, n_test),
    levels=[value_level],
    colors="tab:blue",
    linestyles="dotted",
)
level_set_gp_2 = plt.contour(
    unique(observations_gp_2[:, 0]),
    unique(observations_gp_2[:, 1]),
    observations_gp_2[:, 2].reshape(n_test, n_test),
    levels=[value_level],
    colors="tab:orange",
    linestyles="dotted",
)
plt.clabel(level_set_exact, levels=[value_level], fontsize=10, colors="red")
plt.annotate("Target level set", (-0.2, 0.75), color="red")
plt.annotate(
    "Level set estimated with sequential acquisition", (-1.5, 0.5), color="tab:blue"
)
plt.annotate(
    "Level set estimated with parallel acquisition",
    (-1.5, 0.25),
    color="tab:orange",
)
plt.contour(
    unique(observations[:, 0]),
    unique(observations[:, 1]),
    observations[:, 2].reshape(n_test, n_test),
)
bar = plt.colorbar()
bar.set_label(r"$f(x_1,x_2)$")
plt.scatter(points_1[:, 0], points_1[:, 1], marker="*", label="Sequential")
plt.scatter(points_2[:, 0], points_2[:, 1], marker="*", label="Parallel")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.legend()
plt.show()
# The estimated level set
# estimated with both sequential and batch
# acquisition provide
# a good approximation of the target.
# It can be noticed that
# with parallel acquisition,
# a larger share of learning points
# are used for exploration
# contrary to sequential acquisition.

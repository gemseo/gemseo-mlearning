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
"""# Acquisition criterion."""

from __future__ import annotations

import matplotlib.pyplot as plt
from gemseo import configure
from gemseo import configure_logger
from gemseo import sample_disciplines
from gemseo.mlearning.regression.algos.ot_gpr import OTGaussianProcessRegressor
from gemseo.mlearning.regression.quality.r2_measure import R2Measure
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
# dedicated to exploration
# is illustrated in this example.
# More specifically,
# we aim to test here
# the impact of the choice
# of the acquisition criterion
# used to enrich
# the dataset
# on the active learning procedure.
# The function to approximate is
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
# and three identical initial
# Gaussian process regressors from OpenTURNS:
regressor_1 = OTGaussianProcessRegressor(learning_dataset, trend="quadratic")
regressor_2 = OTGaussianProcessRegressor(learning_dataset, trend="quadratic")
regressor_3 = OTGaussianProcessRegressor(learning_dataset, trend="quadratic")

# %%
# Then,
# we build three active learning algorithms
# to test the impact of the choice
# of the acquisition criterion
# on the active learning procedure.
# They respectively refer to the
# variance (default),
# standard deviation
# and distance criteria.
# All other settings are put to
# their default values.
active_learning_1 = ActiveLearningAlgo("Exploration", input_space, regressor_1)
active_learning_2 = ActiveLearningAlgo(
    "Exploration", input_space, regressor_2, criterion_name="StandardDeviation"
)
active_learning_3 = ActiveLearningAlgo(
    "Exploration", input_space, regressor_3, criterion_name="Distance"
)

active_learning_1.acquire_new_points(discipline, n_samples=20)
active_learning_2.acquire_new_points(discipline, n_samples=20)
active_learning_3.acquire_new_points(discipline, n_samples=20)

# %%
# Finally,
# for the three active learning algorithms,
# we plot the training points,
# alongside the original model.

# Creation of the grid
# and estimation of the different quantities
n_test = 10
observations = sample_disciplines(
    [discipline], input_space, "y", algo_name="OT_FULLFACT", n_samples=n_test**2
).values

# Plotting the contours of the Rosenbrock function
# alongside the learning points.
plt.figure()
plt.contour(
    unique(observations[:, 0]),
    unique(observations[:, 1]),
    observations[:, 2].reshape(n_test, n_test),
)
bar = plt.colorbar()
bar.set_label(r"$f(x_1,x_2)$")
points_1 = active_learning_1.regressor.learning_set.to_numpy()
points_2 = active_learning_2.regressor.learning_set.to_numpy()
points_3 = active_learning_3.regressor.learning_set.to_numpy()
plt.scatter(points_1[:, 0], points_1[:, 1], marker="*", label="Variance")
plt.scatter(points_2[:, 0], points_2[:, 1], marker="*", label="Standard deviation")
plt.scatter(points_3[:, 0], points_3[:, 1], marker="*", label="Distance")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.legend()
plt.show()

# %%
# In this implementation,
# Variance and standard deviation
# criteria provide the same
# learning points,
# which is expected because
# they are identical up to
# a square transformation.
# On the contrary,
# learning points from the distance criterion
# are much different
# and uniformly fills the input space.
# All 3 methods provide a
# surrogate with good accuracy
dataset_test = sample_disciplines(
    [discipline], input_space, "y", algo_name="OT_OPT_LHS", n_samples=50
)
R2_1 = R2Measure(active_learning_1.regressor).compute_test_measure(dataset_test)
R2_2 = R2Measure(active_learning_2.regressor).compute_test_measure(dataset_test)
R2_3 = R2Measure(active_learning_3.regressor).compute_test_measure(dataset_test)
R2_1[0], R2_2[0], R2_3[0]

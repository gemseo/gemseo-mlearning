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
"""# Default settings."""

from __future__ import annotations

from gemseo import configure
from gemseo import configure_logger
from gemseo import sample_disciplines
from gemseo.mlearning.regression.algos.ot_gpr import OTGaussianProcessRegressor
from gemseo.mlearning.regression.quality.r2_measure import R2Measure

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
# is illustrated in this example,
# with all default settings.
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
# and an initial Gaussian process regressor from OpenTURNS:
regressor = OTGaussianProcessRegressor(learning_dataset, trend="quadratic")

# %%
# Then,
# we look for 20 points that will help us
# to improve the overall accuracy
# of the surrogate model.
# By default,
# for this purpose,
# the active learning algorithm looks
# for the point maximizing
# the regressor local variance
# with the help of the SLSQP gradient-based algorithm
# applied in a multistart framework.
active_learning = ActiveLearningAlgo("Exploration", input_space, regressor)
active_learning.acquire_new_points(discipline, 20)

# %%
# Lastly,
# we plot the training points,
# the original model,
# the Gaussian process regressor
# and the variance
# after the last acquisition:
active_learning.plot_acquisition_view(discipline=discipline)

# %%
# The infill criterion targets areas
# where the variance is the highest.
# This occurs in particular on the
# borders of the design space.
# The learning points are not
# uniformly distributed within
# the input space.
# Overall,
# the target surrogate reproduces
# faithfully the Rosenbrock function with a good R²
dataset_test = sample_disciplines(
    [discipline], input_space, "y", algo_name="OT_OPT_LHS", n_samples=50
)
R2 = R2Measure(active_learning.regressor).compute_test_measure(dataset_test)
R2
# Caution:
# the variance is scaled and thus
# does not here directly corresponds
# to the square of the standard deviation.

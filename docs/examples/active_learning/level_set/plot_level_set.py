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
# is illustrated in this example,
# with all default settings.
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
# and an initial Gaussian process regressor from OpenTURNS:
regressor = OTGaussianProcessRegressor(learning_dataset, trend="quadratic")

# %%
# Then,
# we look for 20 points that will help us
# to approximate the level-set to value 300.0.
# By default,
# for this purpose,
# the active learning algorithm looks
# for the point minimizing the U-function
# with the help of the SLSQP algorithm
# applied in a multistart framework.
level_value = 300.0
active_learning = ActiveLearningAlgo(
    "LevelSet", input_space, regressor, output_value=level_value
)
active_learning.acquire_new_points(discipline, 20)

# %%
# Lastly,
# we plot the training points,
# the original model,
# the Gaussian process regressor
# and the U-function
# after the last acquisition:
active_learning.plot_acquisition_view(discipline=discipline)
# It can be seen
# that the learning points
# are distributed around
# the target level set,
# thus approximating it properly.
# After the 25th points,
# the target level set
# is known well enough
# to allow to spend
# computational budget on exploration.

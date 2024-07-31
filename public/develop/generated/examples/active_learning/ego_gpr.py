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
"""# Efficient global optimization using a Gaussian process regressor."""

from __future__ import annotations

from gemseo import configure
from gemseo import configure_logger
from gemseo import sample_disciplines
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.algos.gpr import GaussianProcessRegressor

from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo
from gemseo_mlearning.active_learning.visualization.acquisition_view import (
    AcquisitionView,
)

configure(False, False, True, False, False, False, False)
configure_logger()

# %%
# We consider the Rosenbrock function $f(x,y)=(1-x)^2+100(y-x^2)^2$:
discipline = AnalyticDiscipline({"z": "(1-x)**2+100*(y-x**2)**2"}, name="Rosenbrock")

# %%
# defined over the input space $[-2,2]^2$:
input_space = DesignSpace()
input_space.add_variable("x", l_b=-2, u_b=2, value=1.0)
input_space.add_variable("y", l_b=-2, u_b=2, value=1.0)


# %%
# First,
# we create an initial training dataset using an optimal LHS including 30 samples:
learning_dataset = sample_disciplines([discipline], input_space, "z", 30, "OT_OPT_LHS")

# %%
# and an initial Gaussian process regressor:
gpr = GaussianProcessRegressor(learning_dataset)

# %%
# Thirdly,
# we look for 20 points that will help us get closer to the minimum;
# by default,
# for this purpose,
# the active learning algorithm looks for the point maximizing the expected improvement.
active_learning = ActiveLearningAlgo("Minimum", input_space, gpr)
active_learning.set_acquisition_algorithm("DIFFERENTIAL_EVOLUTION", max_iter=1000)
# active_learning.set_acquisition_algorithm("fullfact", n_samples=30 * 30)
active_learning.acquire_new_points(discipline, 20)

# %%
# Lastly,
# we plot the training points,
# the original model,
# the Gaussian process regressor
# and the expected improvement
# after the last acquisition:
acquisition_view = AcquisitionView(active_learning)
acquisition_view.draw(discipline=discipline)

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
"""# Quantile estimation using a Gaussian process regressor."""

from __future__ import annotations

from gemseo import configure
from gemseo import configure_logger
from gemseo import sample_disciplines
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.algos.gpr import GaussianProcessRegressor
from gemseo.uncertainty.statistics.empirical_statistics import EmpiricalStatistics

from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo

configure(False, False, True, False, False, False, False)
configure_logger()

# %%
# We consider the Rosenbrock function $f(x,y)=(1-x)^2+100(y-x^2)^2$:
discipline = AnalyticDiscipline({"z": "(1-x)**2+100*(y-x**2)**2"}, name="Rosenbrock")

# %%
# defined over the input space $[-2,2]^2$:
input_space = DesignSpace()
input_space.add_variable("x", lower_bound=-2, upper_bound=2, value=1.0)
input_space.add_variable("y", lower_bound=-2, upper_bound=2, value=1.0)

# %%
# with the uncertain space:
uncertain_space = ParameterSpace()
uncertain_space.add_random_variable("x", "OTUniformDistribution", minimum=-2, maximum=2)
uncertain_space.add_random_variable("y", "OTUniformDistribution", minimum=-2, maximum=2)


# %%
# First,
# we create an initial training dataset using an optimal LHS including 30 samples:
learning_dataset = sample_disciplines([discipline], input_space, "z", 10, "OT_OPT_LHS")

# %%
# and an initial Gaussian process regressor:
gpr = GaussianProcessRegressor(learning_dataset)

# %%
# Then,
# we look for 20 points that will help us get closer to the minimum;
# by default,
# for this purpose,
# the active learning algorithm looks for the point maximizing the U-function.
level = 0.35
active_learning = ActiveLearningAlgo(
    "Quantile",
    input_space,
    gpr,
    level=level,
    uncertain_space=uncertain_space,
    # criterion_name="EF",
)
# active_learning.set_acquisition_algorithm("fullfact", n_samples=30 * 30)
# active_learning.set_acquisition_algorithm("NELDER-MEAD")
# active_learning.set_acquisition_algorithm("SLSQP")
active_learning.set_acquisition_algorithm("DIFFERENTIAL_EVOLUTION")
active_learning.acquire_new_points(discipline, 90)

# %%
# Lastly,
# we plot the history of the quantity of interest
active_learning.plot_qoi_history()

# %%
# as well as
# the training points,
# the original model,
# the Gaussian process regressor
# and the U-function
# after the last acquisition:
active_learning.plot_acquisition_view(discipline=discipline)

dataset = sample_disciplines(
    [discipline], uncertain_space, "z", 100000, "OT_MONTE_CARLO"
)
reference_quantile = EmpiricalStatistics(dataset, ["z"]).compute_quantile(level)
print(reference_quantile, active_learning.qoi)

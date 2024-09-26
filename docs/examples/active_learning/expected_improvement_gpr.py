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
"""# Expected improvement using a Gaussian process regressor."""

from __future__ import annotations

import matplotlib.pyplot as plt
from gemseo.algos.design_space import DesignSpace
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.algos.gpr import GaussianProcessRegressor
from numpy import array
from numpy import cos
from numpy import linspace

from gemseo_mlearning.active_learning.acquisition_criteria.maximum.ucb import UCB
from gemseo_mlearning.active_learning.acquisition_criteria.minimum.ei import EI
from gemseo_mlearning.active_learning.acquisition_criteria.minimum.lcb import LCB
from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo
from gemseo_mlearning.active_learning.distributions.kriging_distribution import (
    KrigingDistribution,
)


# %%
# We consider the function $f(x)=10\cos(2x)+15-5x+x^2/50$:
def f(x):
    return (10 * cos(2 * x) + 15 - 5 * x + x**2) / 50


# %%
# defined over the input space $[-3,3]$:
input_space = DesignSpace()
lower_bound = -3.0
upper_bound = 3.0
input_space.add_variable(
    "x", lower_bound=lower_bound, upper_bound=upper_bound, value=1.5
)

# %%
# First,
# we create an initial training dataset:
x_train = array([-2.4, -1.2, 0.0, 1.2, 2.4])
y_train = f(x_train)

training_dataset = IODataset()
training_dataset.add_input_variable("x", x_train)
training_dataset.add_output_variable("y", y_train)

# %%
# and an initial Gaussian process regressor:
gpr = GaussianProcessRegressor(training_dataset)

# %%
# Then,
# we create a Kriging distribution:
kriging_distribution = KrigingDistribution(gpr)
kriging_distribution.learn()

# %%
# as well as three acquisition criteria:
expected_improvement = EI(kriging_distribution)
mean_minus_2sigma = LCB(kriging_distribution, kappa=2.0)
mean_plus_2sigma = UCB(kriging_distribution, kappa=2.0)

# %%
# Thirdly,
# we look for the point that will help us get closer to the minimum;
# by default,
# for this purpose,
# the active learning algorithm looks for the point maximizing the expected improvement.
active_learning = ActiveLearningAlgo("Minimum", input_space, kriging_distribution)
active_learning.set_acquisition_algorithm("TNC")
next_input_data = active_learning.find_next_point()[0]

# %%
# Fourthly,
# we evaluate the different acquisition criteria over a fine grid
# as well as the original function and the Gaussian process regressor:
ei_values = []
predictions = []
mm2s_values = []
mp2s_values = []
x_test = linspace(lower_bound, upper_bound, 200)
y_test = f(x_test)
for x_i in x_test:
    x_i = array([x_i])
    predictions.append(gpr.predict(x_i)[0])
    ei_values.append(expected_improvement.func(x_i)[0])
    mm2s_values.append(mean_minus_2sigma.func(x_i)[0] * mean_minus_2sigma.output_range)
    mp2s_values.append(mean_plus_2sigma.func(x_i)[0] * mean_plus_2sigma.output_range)

# %%
# Lastly,
# we plot the training points,
# the original model,
# the Gaussian process regressor
# and the 95% confidence interval
# on a first sub-plot:
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x_train, y_train, "ro", label="Training points")
ax1.plot(x_test, y_test, "r", label="Original model")
ax1.plot(x_test, predictions, "b", label="GP regressor")
ax1.fill_between(
    x_test, mm2s_values, mp2s_values, color="b", alpha=0.1, label="CI(95%)"
)
ax1.grid()
# %%
# and the expected improvement on a second plots:
ax1.legend(loc="upper right")
ax1.axvline(x=next_input_data)
ax2.plot(x_test, ei_values, "r", label="EI")
ax2.axvline(x=next_input_data)
ax2.legend()
ax2.grid()
plt.show()

# %%
# The vertical blue line indicates the point maximizing the expected improvement.

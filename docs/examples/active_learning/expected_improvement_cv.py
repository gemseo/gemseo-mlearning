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
"""# Expected improvement based on cross-validation."""

from __future__ import annotations

import matplotlib.pyplot as plt
from gemseo.algos.design_space import DesignSpace
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.algos.rbf import RBFRegressor
from numpy import array
from numpy import cos
from numpy import linspace

from gemseo_mlearning.active_learning.acquisition_criteria.maximum.ucb import UCB
from gemseo_mlearning.active_learning.acquisition_criteria.minimum.ei import EI
from gemseo_mlearning.active_learning.acquisition_criteria.minimum.lcb import LCB
from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo
from gemseo_mlearning.active_learning.distributions.regressor_distribution import (
    RegressorDistribution,
)

n_test = 200
x_l = -3.0
x_u = 3.0


# %%
# ## Initial learning dataset
def f(x):
    return (10 * cos(2 * x) + 15 - 5 * x + x**2) / 50


x_train = array([-2.4, -1.2, 0.0, 1.2, 2.4])
y_train = f(x_train)

dataset = IODataset()
dataset.add_input_variable("x", x_train)
dataset.add_output_variable("y", y_train)

# %%
# ## Initial surrogate model
algo = RBFRegressor(dataset)
algo.learn()

# %%
# ## Create MLAlgoSampler
distribution = RegressorDistribution(algo, bootstrap=False, loo=True)
distribution.learn()

# %%
# ## Filling objectives
ego = EI(distribution)
lower = LCB(distribution, kappa=2.0)
upper = UCB(distribution, kappa=2.0)

# %%
# ## Find next training point
space = DesignSpace()
space.add_variable("x", l_b=x_l, u_b=x_u, value=1.5)

acquisition = ActiveLearningAlgo("Minimum", space, distribution)
acquisition.set_acquisition_algorithm("fullfact")
opt = acquisition.find_next_point()

# %%
# ## Evaluation of discipline, surrogate model and expected improvement
x_test = linspace(x_l, x_u, n_test)
ego_data = []
surr_data = []
lower_data = []
upper_data = []
y_test = f(x_test)
for x_i in x_test:
    surr_data.append(algo.predict(array([x_i]))[0])
    ego_data.append(ego(array([x_i]))[0])
    lower_data.append(lower(array([x_i]))[0] * lower.output_range)
    upper_data.append(upper(array([x_i]))[0] * upper.output_range)
ego_data = array(ego_data)
lower_data = array(lower_data)
upper_data = array(upper_data)

disc_data = IODataset()
disc_data.add_input_variable("x", x_test)
disc_data.add_output_variable("y", y_test)

# %%
# ## Plotting
fig, ax = plt.subplots(2, 1)
for algo_b in distribution.algos:
    algo_data = [algo_b.predict(array([x_i])) for x_i in x_test]
    ax[0].plot(x_test, algo_data, "gray", alpha=0.2)
ax[0].plot(
    x_train, dataset.get_view(variable_names="y").to_numpy(), "ro", label="training"
)
ax[0].plot(
    x_test, disc_data.get_view(variable_names="y").to_numpy(), "r", label="original"
)
ax[0].plot(x_test, surr_data, "b", label="surrogate")
ax[0].fill_between(
    x_test, lower_data, upper_data, color="b", alpha=0.1, label="CI(95%)"
)
ax[0].legend(loc="upper right")
ax[0].axvline(x=opt[0])
ax[1].plot(x_test, ego_data, "r", label="EGO")
ax[1].axvline(x=opt[0])
ax[1].legend()
plt.show()

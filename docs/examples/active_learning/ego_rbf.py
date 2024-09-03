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
"""# EGO based on resampling."""

from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from gemseo import configure
from gemseo import configure_logger
from gemseo import sample_disciplines
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.algos.rbf import RBFRegressor
from numpy import array
from numpy import linspace
from numpy import zeros

from gemseo_mlearning.active_learning.acquisition_criteria.minimum.ei import EI
from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo
from gemseo_mlearning.active_learning.distributions.regressor_distribution import (
    RegressorDistribution,
)

configure(False, False, True, False, False, False, False)
configure_logger()

n_test = 20

# %%
# Definition of the discipline
discipline = AnalyticDiscipline({"z": "(1-x)**2+100*(y-x**2)**2"}, name="Rosenbrock")

# %%
# ## Definition of the input space
input_space = DesignSpace()
input_space.add_variable("x", l_b=-2, u_b=2, value=1.0)
input_space.add_variable("y", l_b=-2, u_b=2, value=1.0)


# %%
# ## Initial surrogate model
learning_dataset = sample_disciplines([discipline], input_space, "z", 30, "OT_OPT_LHS")
algo = RBFRegressor(learning_dataset)

# %%
# ## Universal distribution for this surrogate model
distribution = RegressorDistribution(algo, bootstrap=False)
distribution.learn()

# %%
# ## Data acquisition to improve the surrogate model
acquisition = ActiveLearningAlgo("Minimum", input_space, distribution)
acquisition.set_acquisition_algorithm("fullfact", n_samples=1000)
acquisition.acquire_new_points(discipline, 20)

opt = distribution.algo.learning_set.get_view(variable_names=["x", "y"]).to_numpy()
opt_x = opt[:, 0]
opt_y = opt[:, 1]


# %%
# ## Evaluation of discipline and expected improvement
crit = EI(distribution)
x_test = linspace(-2, 2, n_test)
disc_data = zeros((n_test, n_test))
crit_data = zeros((n_test, n_test))
surr_data = zeros((n_test, n_test))
for i in range(n_test):
    for j in range(n_test):
        xij = array([x_test[j], x_test[i]])
        input_data = {"x": array([xij[0]]), "y": array([xij[1]])}
        disc_data[i, j] = discipline.execute(input_data)["z"][0]
        crit_data[i, j] = crit.func(xij)
        surr_data[i, j] = algo.predict(xij)[0]

# %%
# ## Plotting
train = learning_dataset.get_view(variable_names=["x", "y"]).to_numpy()
x_train = train[:, 0]
y_train = train[:, 1]
fig = plt.figure(constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
axes = [[None, None], [None, None]]
titles = [["Discipline", "Infill criterion"], ["Surrogate model", None]]
data = [[disc_data, crit_data], [surr_data, None]]
for i in range(2):
    for j in range(2):
        if [i, j] != [1, 1]:
            axes[i][j] = fig.add_subplot(spec[i, j])
            axes[i][j].contourf(x_test, x_test, data[i][j])
            axes[i][j].axhline(1, color="white", alpha=0.5)
            axes[i][j].axvline(1, color="white", alpha=0.5)
            axes[i][j].plot(x_train, y_train, "w+", ms=1)
            for index, _ in enumerate(opt_x):
                axes[i][j].plot(opt_x[index], opt_y[index], "wo", ms=1)
                axes[i][j].annotate(
                    index + 1,
                    (opt_x[index] + 0.05, opt_y[index] + 0.05),
                    color="white",
                )
            axes[i][j].set_title(titles[i][j])
plt.show()

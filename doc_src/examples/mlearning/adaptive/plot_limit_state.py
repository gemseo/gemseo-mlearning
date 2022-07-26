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
# Copyright (C) 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Limit state based on resampling
===============================
"""
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from gemseo.algos.design_space import DesignSpace
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.rbf import RBFRegressor
from gemseo_mlearning.adaptive.acquisition import MLDataAcquisition
from gemseo_mlearning.adaptive.criteria.quantile.criterion import Quantile
from gemseo_mlearning.adaptive.distributions.regressor_distribution import (
    RegressorDistribution,
)
from numpy import array
from numpy import linspace
from numpy import zeros

l_b = -2
u_b = 2

n_test = 20
level = 0.8

##############################################################################
# Definition of the discipline
# ----------------------------
discipline = AnalyticDiscipline({"z": "(1-x)**2+100*(y-x**2)**2"}, name="Rosenbrock")

##############################################################################
# Definition of the input space
# -----------------------------
input_space = DesignSpace()
input_space.add_variable("x", l_b=-2, u_b=2, value=1.0)
input_space.add_variable("y", l_b=-2, u_b=2, value=1.0)

##############################################################################
# Initial surrogate model
# -----------------------
scenario = DOEScenario([discipline], "DisciplinaryOpt", "z", input_space)
scenario.execute({"algo": "OT_OPT_LHS", "n_samples": 30})
opt_problem = scenario.formulation.opt_problem
learning_dataset = opt_problem.export_to_dataset("learning", opt_naming=False)

algo = RBFRegressor(learning_dataset)

##############################################################################
# Universal distribution for this surrogate model
# -----------------------------------------------
distribution = RegressorDistribution(algo, bootstrap=False)
distribution.learn()

##############################################################################
# Data acquisition to improve the surrogate model
# -----------------------------------------------
acquisition = MLDataAcquisition("Quantile", input_space, distribution, level=level)
acquisition.set_acquisition_algorithm("fullfact", n_samples=1000)
acquisition.update_algo(discipline, 20)

points = distribution.algo.learning_set[["x", "y"]]
points_x = points["x"]
points_y = points["y"]

##############################################################################
# Evaluation of discipline and expected improvement
# -------------------------------------------------
crit = Quantile(distribution, level)
x_test = linspace(l_b, u_b, n_test)
disc_data = zeros((n_test, n_test))
crit_data = zeros((n_test, n_test))
surr_data = zeros((n_test, n_test))
for i in range(n_test):
    for j in range(n_test):
        xij = array([x_test[j], x_test[i]])
        input_data = {"x": array([xij[0]]), "y": array([xij[1]])}
        disc_data[i, j] = discipline.execute(input_data)["z"][0]
        crit_data[i, j] = crit(xij)
        surr_data[i, j] = algo.predict(xij)[0]

##############################################################################
# Plotting
# --------
train = learning_dataset[["x", "y"]]
x_train = train["x"]
y_train = train["y"]
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
            axes[i][j].plot(x_train[:, 0], y_train[:, 0], "w+", ms=1)
            for index, _ in enumerate(points_x):
                axes[i][j].plot(points_x[index, 0], points_y[index, 0], "wo", ms=1)
                axes[i][j].annotate(
                    index + 1,
                    (points_x[index, 0] + 0.05, points_y[index, 0] + 0.05),
                    color="white",
                )
            axes[i][j].set_title(titles[i][j])
plt.show()

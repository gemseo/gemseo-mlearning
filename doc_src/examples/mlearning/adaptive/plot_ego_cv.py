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
Expected improvement based on cross-validation
==============================================
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.rbf import RBFRegressor
from numpy import array
from numpy import linspace

from gemseo_mlearning.adaptive.acquisition import MLDataAcquisition
from gemseo_mlearning.adaptive.criteria.mean_std.criterion import MeanSigma
from gemseo_mlearning.adaptive.criteria.optimum.criterion import ExpectedImprovement
from gemseo_mlearning.adaptive.distributions.regressor_distribution import (
    RegressorDistribution,
)

n_test = 200
x_l = -3.0
x_u = 3.0

##############################################################################
# Initial learning dataset
# ------------------------
discipline = AnalyticDiscipline({"y": "(10*cos(2*x)+15-5*x+x**2)/50"})
discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
x_train = [-2.4, -1.2, 0.0, 1.2, 2.4]
for x_i in x_train:
    discipline.execute({"x": array([x_i])})
dataset = discipline.cache.export_to_dataset()

##############################################################################
# Initial surrogate model
# -----------------------
algo = RBFRegressor(dataset)
algo.learn()

##############################################################################
# Create MLAlgoSampler
# --------------------
distribution = RegressorDistribution(algo, bootstrap=False, loo=True)
distribution.learn()

##############################################################################
# Filling objectives
# ------------------
ego = ExpectedImprovement(distribution)
lower = MeanSigma(distribution, -2.0)
upper = MeanSigma(distribution, 2.0)

##############################################################################
# Find next training point
# ------------------------
space = DesignSpace()
space.add_variable("x", l_b=x_l, u_b=x_u, value=1.5)

acquisition = MLDataAcquisition("ExpectedImprovement", space, distribution)
acquisition.set_acquisition_algorithm("fullfact")
opt = acquisition.compute_next_input_data()

##############################################################################
# Evaluation of discipline, surrogate model and expected improvement
# ------------------------------------------------------------------
discipline.cache.clear()
x_test = linspace(x_l, x_u, n_test)
ego_data = []
surr_data = []
lower_data = []
upper_data = []
for x_i in x_test:
    discipline.execute({"x": array([x_i])})
    surr_data.append(algo.predict(array([x_i]))[0])
    ego_data.append(ego(array([x_i]))[0])
    lower_data.append(lower(array([x_i]))[0] * lower.output_range)
    upper_data.append(upper(array([x_i]))[0] * upper.output_range)
ego_data = array(ego_data)
lower_data = array(lower_data)
upper_data = array(upper_data)

disc_data = discipline.cache.export_to_dataset()

##############################################################################
# Plotting
# --------
fig, ax = plt.subplots(2, 1)
for algo_b in distribution.algos:
    algo_data = [algo_b.predict(array([x_i])) for x_i in x_test]
    ax[0].plot(x_test, algo_data, "gray", alpha=0.2)
ax[0].plot(
    x_train, dataset.get_view(variable_names="y").to_numpy(), "ro", label="training"
)
ax[0].plot(x_test, disc_data["y"], "r", label="original")
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

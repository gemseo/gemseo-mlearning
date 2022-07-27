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
#
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Cross-validation vs bootstrap
=============================
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.rbf import RBFRegressor
from gemseo_mlearning.adaptive.acquisition import MLDataAcquisition
from gemseo_mlearning.adaptive.criteria.mean_std.criterion import MeanSigma
from gemseo_mlearning.adaptive.criteria.optimum.criterion import (
    ExpectedImprovement,
)
from gemseo_mlearning.adaptive.distributions.regressor_distribution import (
    RegressorDistribution,
)
from numpy import array
from numpy import linspace

n_test = 200
x_l = -3.0
x_u = 3.0

# %%
# Initial learning dataset
# ------------------------
discipline = AnalyticDiscipline({"y": "(10*cos(2*x)+15-5*x+x**2)/50"})
discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
x_train = [-2.4, -1.2, 0.0, 1.2, 2.4]
for x_i in x_train:
    discipline.execute({"x": array([x_i])})
dataset = discipline.cache.export_to_dataset()

ax = [[None, None], [None, None]]
ax[0][0] = plt.subplot(221)
ax[0][1] = plt.subplot(222, sharey=ax[0][0])
ax[1][0] = plt.subplot(223)
ax[1][1] = plt.subplot(224, sharey=ax[1][0])

# %%
# Active learning
# ---------------
# We compare the bootstrap and leave-one-out methods
# in search of a new point to learn
# in order to estimate the minimum of the discipline
for index, bootstrap in enumerate([False, True]):

    # Train a RBF regression model
    algo = RBFRegressor(dataset)
    algo.learn()

    # Build a regressor distribution
    distribution = RegressorDistribution(algo, bootstrap=bootstrap, loo=not bootstrap)
    distribution.learn()

    # Define the expected improvement measure
    ego = ExpectedImprovement(distribution)

    # Define confidence bounds, equal to mean +/- 2*sigma
    lower = MeanSigma(distribution, -2.0)
    upper = MeanSigma(distribution, 2.0)

    # Define the input_space
    input_space = DesignSpace()
    input_space.add_variable("x", l_b=x_l, u_b=x_u, value=1.5)

    # Define the data acquisition process
    acquisition = MLDataAcquisition("ExpectedImprovement", input_space, distribution)
    acquisition.set_acquisition_algorithm("fullfact")

    # Compute the next input data
    opt = acquisition.compute_next_input_data()

    # Plot the results
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

    disc_data = discipline.cache.export_to_dataset()

    for algo in distribution.algos:
        algo_data = [algo.predict(array([x_i])) for x_i in x_test]
        ax[0][index].plot(x_test, algo_data, "gray", alpha=0.2)

    ax[0][index].plot(x_train, dataset["y"], "ro", label="training")
    ax[0][index].plot(x_test, disc_data["y"], "r", label="original")
    ax[0][index].plot(x_test, surr_data, "b", label="surrogate")
    ax[0][index].fill_between(
        x_test,
        array(lower_data),
        array(upper_data),
        color="b",
        alpha=0.1,
        label="CI(95%)",
    )
    ax[0][index].legend(loc="upper right")
    ax[0][index].axvline(x=opt[0])
    ax[1][index].plot(x_test, array(ego_data), "r", label="EGO")
    ax[1][index].axvline(x=opt[0])
    ax[1][index].legend()
plt.show()

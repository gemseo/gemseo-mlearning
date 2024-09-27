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
"""# SMT's surrogate model.

The [surrogate modeling toolbox (SMT)](../../../user_guide/regression/smt.md)
is an open-source Python package for surrogate modeling.
The [SMTRegressor][gemseo_mlearning.regression.smt_regressor.SMTRegressor] class
allows you to use any SMT's surrogate model in your GEMSEO processes.

In this example,
we will approximate the
[Rosenbrock function][@molga2005test]
$$f(x,y) = (1-x)^2 + 100(y-x^2)^2$$
over the domain $[-2,2]^2$.
"""

from gemseo import compute_doe
from gemseo import sample_disciplines
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.quality.r2_measure import R2Measure
from gemseo.post.dataset.zvsxy import ZvsXY

from gemseo_mlearning.problems.rosenbrock.rosenbrock_discipline import (
    RosenbrockDiscipline,
)
from gemseo_mlearning.problems.rosenbrock.rosenbrock_space import RosenbrockSpace
from gemseo_mlearning.regression.smt_regressor import SMTRegressor

# %%
# First,
# we create the Rosenbrock discipline:
discipline = RosenbrockDiscipline()

# %%
# and the input space:
input_space = RosenbrockSpace()

# %%
# Then,
# we use an optimized Latin hypercube sampling (LHS) technique
# to generate 20 samples:
training_data = sample_disciplines([discipline], input_space, "y", 20, "OT_OPT_LHS")

# %%
# From this learning dataset,
# we train an [SMTRegressor][gemseo_mlearning.regression.smt_regressor.SMTRegressor]
# based on the
# [SMT's RBF surrogate model](https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/rbf.html)
# with the basis function scaling parameter `d_0` set to 2.0 (instead of 1.0):
surrogate_model = SMTRegressor(training_data, "RBF", d0=2)
surrogate_model.learn()

# %%
# Finally,
# we assess its quality:
test_data = sample_disciplines([discipline], input_space, "y", 1000, "OT_MONTE_CARLO")
r2 = R2Measure(surrogate_model)
f"Learning R2: {r2.compute_learning_measure()[0]}; test R2: {r2.compute_test_measure(test_data)[0]}"

# %%
# see how good it is with its R2 close to 1 on the test dataset,
# and plot its output over a 20x20 grid:
input_data = compute_doe(input_space, "fullfact", 400)
output_data = surrogate_model.predict(input_data)
predictions = IODataset()
predictions.add_input_group(input_data, variable_names=["x1", "x2"])
predictions.add_output_group(output_data, variable_names=["y"])

plot = ZvsXY(predictions, "x1", "x2", "y", other_datasets=(training_data,))
plot.color = "white"
plot.colormap = "viridis"
plot.execute(save=False, show=True)

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
"""# Surrogate-based optimization using in-house features."""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import execute_algo
from gemseo.post.dataset.zvsxy import ZvsXY
from gemseo.problems.dataset.rosenbrock import create_rosenbrock_dataset
from gemseo.problems.optimization.rosenbrock import Rosenbrock

configure_logger()

# %%
# In this example,
# we seek to minimize the Rosenbrock function $f(x,y)=(1-x)^2+100(y-x^2)^2$
# over the design space $[-2,2]^2$.
# First,
# we instantiate the problem with $(0, 0)$ as initial guess:
problem = Rosenbrock()

# %%
# Then,
# we minimize the Rosenbrock function using:
#
# - the `"SBO"` algorithm,
# - a maximum number of 40 evaluations,
#   including the initial one at the center of the design space
#   (this first point is common to all optimization algorithms)
#   and the initial training dataset,
# - its default settings,
#   namely:
#
#   - the expected improvement as acquisition criterion,
#   - 1 point acquired at a time,
#   - the `GaussianProcessRegressor` based on scikit-learn,
#   - 10 initial training points
#     based on an optimized latin hypercube sampling (LHS) technique,
#   - a multi-start local optimization of the acquisition criterion
#     from 50 start points with a limit of 20 iterations per local optimization.
execute_algo(problem, "SBO", max_iter=40)

# %%
# We can see
# that the solution is close to the theoretical one $(x^*,f^*)=((1,1),0)$.
#
# We can also visualize all the evaluations
# and note that most of the points have been added in the valley as expected:
optimization_history = problem.to_dataset()

initial_point = optimization_history[0:1]
initial_point.name = "Initial point"

initial_training_points = optimization_history[1:12]
initial_training_points.name = "Initial training points"

acquired_points = optimization_history[12:]
acquired_points.name = "Acquired points"

visualization = ZvsXY(
    create_rosenbrock_dataset(900),
    ("x", 0),
    ("x", 1),
    "rosen",
    fill=False,
    other_datasets=(initial_point, initial_training_points, acquired_points),
)
visualization.execute(save=False, show=True)

# %%
# Lastly,
# we can compare the solution to the one obtained with COBYLA,
# which is another popular gradient-free optimization algorithm:
execute_algo(Rosenbrock(), "NLOPT_COBYLA", max_iter=40)

# %%
# and conclude that for this problem and this initial guess,
# the surrogate-based algorithm is better than COBYLA.

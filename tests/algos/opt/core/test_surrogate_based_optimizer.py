# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Tests for the surrogate-based optimizer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.optimization.rastrigin import Rastrigin
from pandas._testing import assert_frame_equal

from gemseo_mlearning.algos.opt.core.surrogate_based_optimizer import (
    SurrogateBasedOptimizer,
)


@pytest.mark.parametrize(
    ("regression_algorithm", "regression_options"),
    [
        ("GaussianProcessRegressor", {}),
        ("OTGaussianProcessRegressor", {"use_hmat": False}),
    ],
)
def test_all_acquisitions_made(regression_algorithm, regression_options):
    """Check the execution of the surrogate-based optimizer with all acquisitions."""
    assert (
        SurrogateBasedOptimizer(
            Rastrigin(),
            "fullfact",
            5,
            regression_algorithm=regression_algorithm,
            regression_options=regression_options,
        ).execute(1)
        == "All the data acquisitions have been made."
    )


def test_known_acquired_input_data():
    """Check the termination when the acquired input data is already known."""
    space = DesignSpace()
    space.add_variable("x", l_b=0, u_b=1)
    problem = OptimizationProblem(space)
    problem.objective = MDOFunction(lambda _: 0, "f")
    assert (
        SurrogateBasedOptimizer(
            problem,
            "DIFFERENTIAL_EVOLUTION",
            2,
            regression_algorithm="LinearRegressor",
        ).execute(1)
        == "The acquired input data is already known."
    )


def test_convergence_on_rastrigin():
    """Check the surrogate-based optimizer on Rastrigin's function."""

    def listener(x):
        return

    problem = Rastrigin()
    problem.database.add_store_listener(listener)
    problem.database.add_store_listener = MagicMock()
    SurrogateBasedOptimizer(
        problem,
        "DIFFERENTIAL_EVOLUTION",
        doe_size=20,
        acquisition_options={"max_iter": 1000, "popsize": 50, "seed": 1},
    ).execute(5)
    assert problem.optimum.objective < 0.1

    # Check that the optimizer resets the listener after the sub-algo has removed it.
    problem.database.add_store_listener.assert_called()


def test_stratified_algorithm():
    """Check the use of a stratified algorithm for the initial sampling."""
    assert (
        SurrogateBasedOptimizer(
            Rastrigin(),
            "DIFFERENTIAL_EVOLUTION",
            doe_algorithm="OT_AXIAL",
            doe_options={"centers": [0.5, 0.5], "levels": [0.1, 0.2]},
        ).execute(1)
        == "All the data acquisitions have been made."
    )


def test_ml_regression_algo_instance(regression_algorithm):
    """Check the execution of the surrogate-based optimizer with an
    BaseMLRegressionAlgo."""
    optimizer = SurrogateBasedOptimizer(
        Rastrigin(), "fullfact", regression_algorithm=regression_algorithm
    )
    optimizer.execute(1)
    dataset = optimizer._SurrogateBasedOptimizer__dataset

    optimizer = SurrogateBasedOptimizer(
        Rastrigin(),
        "fullfact",
        doe_size=5,
        doe_algorithm="OT_SOBOL",
        regression_algorithm="OTGaussianProcessRegressor",
    )
    optimizer.execute(1)
    assert_frame_equal(optimizer._SurrogateBasedOptimizer__dataset, dataset)

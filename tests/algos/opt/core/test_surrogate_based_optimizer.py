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
from gemseo.core.function.array_function import ArrayFunction
from gemseo.doe import CustomDOE_Settings
from gemseo.doe import OT_AXIAL_Settings
from gemseo.doe import OT_OPT_LHS_Settings
from gemseo.doe import OT_SOBOL_Settings
from gemseo.doe import PYDOE_FULLFACT_Settings
from gemseo.machine_learning.regression.model.gpr_settings import (
    GaussianProcessRegressor_Settings,
)
from gemseo.machine_learning.regression.model.linreg_settings import (
    LinearRegressor_Settings,
)
from gemseo.machine_learning.regression.model.ot_gpr_settings import (
    OTGaussianProcessRegressor_Settings,
)
from gemseo.optimization import DIFFERENTIAL_EVOLUTION_Settings
from gemseo.optimization import OptimizationProblem
from gemseo.problem.optimization.rastrigin import Rastrigin
from gemseo.space import DesignSpace
from numpy import array
from pandas._testing import assert_frame_equal

from gemseo_mlearning.algos.opt.core.surrogate_based_optimizer import (
    SurrogateBasedOptimizer,
)


@pytest.mark.parametrize(
    "regressor",
    [
        GaussianProcessRegressor_Settings(),
        OTGaussianProcessRegressor_Settings(use_hmat=False),
    ],
)
def test_all_acquisitions_made(regressor):
    """Check the execution of the surrogate-based optimizer with all acquisitions."""
    assert (
        SurrogateBasedOptimizer(
            Rastrigin(),
            PYDOE_FULLFACT_Settings(n_samples=10),
            OT_OPT_LHS_Settings(n_samples=5),
            regressor=regressor,
        ).execute(1)
        == "All the data acquisitions have been made."
    )


def test_known_acquired_input_data():
    """Check the termination when the acquired input data is already known."""
    space = DesignSpace()
    space.add_variable("x", lower_bound=0, upper_bound=1)
    problem = OptimizationProblem(space)
    problem.objective = ArrayFunction(lambda _: 0, "f")
    assert (
        SurrogateBasedOptimizer(
            problem,
            CustomDOE_Settings(samples=array([[0.0]])),
            OT_OPT_LHS_Settings(n_samples=2),
            regressor=LinearRegressor_Settings(),
        ).execute(2)
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
        DIFFERENTIAL_EVOLUTION_Settings(max_iter=1000, popsize=50, seed=1),
        OT_OPT_LHS_Settings(n_samples=20),
    ).execute(5)
    assert problem.optimum.objective < 0.12

    # Check that the optimizer resets the listener after the sub-algo has removed it.
    problem.database.add_store_listener.assert_called()


def test_stratified_algorithm():
    """Check the use of a stratified algorithm for the initial sampling."""
    assert (
        SurrogateBasedOptimizer(
            Rastrigin(),
            DIFFERENTIAL_EVOLUTION_Settings(max_iter=10),
            OT_AXIAL_Settings(centers=[0.5, 0.5], levels=[0.1, 0.2]),
        ).execute(1)
        == "All the data acquisitions have been made."
    )


@pytest.mark.parametrize("kwargs", [{}, {"transformer": {"inputs": "MinMaxScaler"}}])
def test_ml_regression_algo_instance(regressor, kwargs):
    """Check the execution of the surrogate-based optimizer with an
    BaseMLRegressionAlgo.
    """
    optimizer = SurrogateBasedOptimizer(
        Rastrigin(),
        CustomDOE_Settings(samples=array([[0.03, 0.03]])),
        regressor=regressor,
    )
    optimizer.execute(1)
    dataset = optimizer._SurrogateBasedOptimizer__dataset

    optimizer = SurrogateBasedOptimizer(
        Rastrigin(),
        CustomDOE_Settings(samples=array([[0.03, 0.03]])),
        OT_SOBOL_Settings(n_samples=5),
        regressor=OTGaussianProcessRegressor_Settings(**kwargs),
    )
    optimizer.execute(1)
    assert_frame_equal(optimizer._SurrogateBasedOptimizer__dataset, dataset)

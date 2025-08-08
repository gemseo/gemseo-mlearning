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
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from gemseo.algos.doe.scipy.scipy_doe import SciPyDOE
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.datasets.io_dataset import IODataset
from gemseo.problems.optimization.rosenbrock import Rosenbrock
from numpy import array
from numpy import atleast_2d
from numpy import concatenate
from numpy import full
from numpy import hstack
from numpy import linspace
from numpy import newaxis
from numpy.testing import assert_almost_equal
from smt.surrogate_models import GEKPLS
from smt.surrogate_models import RBF

from gemseo_mlearning.regression.smt_regressor import _NAMES_TO_CLASSES
from gemseo_mlearning.regression.smt_regressor import SMTRegressor
from gemseo_mlearning.regression.smt_regressor_settings import SMT_Regressor_Settings
from gemseo_mlearning.regression.smt_regressor_settings import SMTRegressorSettings


@pytest.mark.parametrize("name", ["RBF", "KRG"])
def test_available_models(name):
    """Check the SMT's surrogate models that can be instantiated."""
    assert name in _NAMES_TO_CLASSES


if TYPE_CHECKING:
    from gemseo.typing import RealArray


@pytest.fixture(scope="module", params=[False, True])
def io_data(request) -> tuple[RealArray, RealArray]:
    """Input and output data."""
    if request.param:
        x = linspace(-1, 1, 10)[:, newaxis]
        y = x**2
    else:
        x = linspace(-1, 1, 10).reshape((5, 2))
        y = hstack(((x[:, [0]] - x[:, [1]]) ** 2, (x[:, [0]] + x[:, [1]]) ** 2))

    return x, y


@pytest.fixture(scope="module")
def regressor_and_model(io_data) -> tuple[SMTRegressor, RBF]:
    """The SMT-based regressor and the corresponding SMT model."""
    input_data, output_data = io_data

    dataset = IODataset()
    dataset.add_input_group(input_data)
    dataset.add_output_group(output_data)

    regressor = SMTRegressor(dataset, model_class_name="RBF", parameters={"d0": 2})
    regressor.learn()

    model = RBF(d0=2)
    model.set_training_values(input_data, output_data)
    model.train()
    return regressor, model


@pytest.fixture(scope="module", params=[False, True])
def input_value(regressor_and_model, request) -> RealArray:
    """The input value."""
    input_value_ = full((regressor_and_model[0].output_dimension,), 0.25)
    if request.param:
        return input_value_[newaxis, :]

    return input_value_


def test_smt_regression_model_predict(regressor_and_model, input_value):
    """Check SMTRegressor.predict method."""
    regressor, model = regressor_and_model
    expected = model.predict_values(atleast_2d(input_value))
    if input_value.ndim == 1:
        expected = expected[0]

    assert_almost_equal(regressor.predict(input_value), expected)


def test_smt_regression_model_predict_jacobian(regressor_and_model, input_value):
    """Check SMTRegressor.predict_jacobian method."""
    regressor, model = regressor_and_model
    expected = model.predict_derivatives(atleast_2d(input_value), 0)[..., newaxis]
    if regressor.output_dimension == 2:
        expected = concatenate(
            (
                expected,
                model.predict_derivatives(atleast_2d(input_value), 1)[..., newaxis],
            ),
            axis=2,
        )

    if input_value.ndim == 1:
        expected = expected[0]

    assert_almost_equal(regressor.predict_jacobian(input_value), expected)


def test_gradient_enhanced_smt_regressor():
    """Check that gradient-enhanced SMT surrogate models use the gradient samples."""

    def compute_sum_jacobian(input_value: RealArray) -> RealArray:
        """Compute the Jacobian of the sum function at a given point.

        Args:
            input_value: The input value.

        Returns:
            The Jacobian.
        """
        return array([[1.0, 1.0]])

    problem = Rosenbrock()
    problem.add_observable(MDOFunction(sum, "sum", jac=compute_sum_jacobian))
    SciPyDOE("LHS").execute(problem, eval_jac=True, n_samples=30)
    dataset = problem.to_dataset(opt_naming=False, export_gradients=True)

    regressor = SMTRegressor(dataset, model_class_name="GEKPLS")

    # Check that the training does not raise an exception
    regressor.learn()

    # Check the default values
    algo = regressor.algo
    assert isinstance(algo, GEKPLS)
    assert algo.options["n_comp"] == 2

    # Check that custom values can be passed
    regressor = SMTRegressor(
        dataset, model_class_name="GEKPLS", parameters={"n_comp": 5}
    )
    assert regressor.algo.options["n_comp"] == 5

    # Check that passing a gradient-free IODataset raises a ValueError
    x = linspace(-1, 1, 10)[:, newaxis]
    y = x**2
    dataset = IODataset()
    dataset.add_input_group(x)
    dataset.add_output_group(y)

    regressor = SMTRegressor(dataset, model_class_name="GEKPLS")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "GEKPLS did not found gradient samples in the training dataset."
        ),
    ):
        regressor.learn()


def test_alias():
    """Verify that SMTRegressorSettings is an alias of SMT_Regressor_Settings."""
    assert SMTRegressorSettings == SMT_Regressor_Settings

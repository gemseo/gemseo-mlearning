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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from operator import eq

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.datasets.io_dataset import IODataset
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.linreg import LinearRegressor
from numpy import array
from numpy import ndarray

from gemseo_mlearning.adaptive.acquisition import MLDataAcquisition
from gemseo_mlearning.adaptive.distributions.regressor_distribution import (
    RegressorDistribution,
)


@pytest.fixture(scope="module")
def dataset() -> IODataset:
    """A learning dataset."""
    dataset = IODataset()
    dataset.add_variable(
        "x", array([0.0, 1.0])[:, None], group_name=dataset.INPUT_GROUP
    )
    dataset.add_variable(
        "y",
        array([1.0, 2.0])[:, None],
        group_name=dataset.OUTPUT_GROUP,
    )
    return dataset


@pytest.fixture(scope="module")
def algo_distribution_for_update(dataset) -> RegressorDistribution:
    """The distribution of a linear regression model to be updated."""
    distribution = RegressorDistribution(LinearRegressor(dataset), size=3)
    distribution.learn()
    return distribution


@pytest.fixture(scope="module")
def algo_distribution(dataset) -> RegressorDistribution:
    """The distribution of a linear regression model."""
    distribution = RegressorDistribution(LinearRegressor(dataset), size=3)
    distribution.learn()
    return distribution


@pytest.fixture(scope="module")
def input_space() -> DesignSpace:
    """The input space used to acquire new points."""
    space = DesignSpace()
    space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)
    return space


def test_init(algo_distribution, input_space):
    """Check the initialization of the MLDataAcquisition."""
    acquisition = MLDataAcquisition(
        "ExpectedImprovement", input_space, algo_distribution
    )
    assert acquisition._MLDataAcquisition__algo_name == acquisition.default_algo_name
    assert (
        acquisition._MLDataAcquisition__algo_options == acquisition.default_opt_options
    )
    assert acquisition._MLDataAcquisition__problem.design_space.variable_names == ["x"]
    assert acquisition._MLDataAcquisition__criterion == "ExpectedImprovement"
    assert acquisition._MLDataAcquisition__input_space == input_space
    assert acquisition._MLDataAcquisition__distribution == algo_distribution
    assert acquisition._MLDataAcquisition__algo.algo_name == "NLOPT_COBYLA"


def test_init_with_bad_output_dimension(input_space):
    """Check MLDataAcquisition initialization raising errors.

    The initialization should raise a NotImplementedError when the output dimension of
    the algorithm is greater than 1.
    """
    dataset = IODataset()
    dataset.add_variable("x", array([[0.0], [1.0]]), group_name=dataset.INPUT_GROUP)
    dataset.add_variable(
        "y",
        array([[1.0, 1.0], [2.0, 2.0]]),
        group_name=dataset.OUTPUT_GROUP,
    )
    algo = LinearRegressor(dataset)
    algo.learn()
    algo_distribution = RegressorDistribution(algo, size=3)
    with pytest.raises(
        NotImplementedError, match="MLDataAcquisition works only with scalar output."
    ):
        MLDataAcquisition("ExpectedImprovement", input_space, algo_distribution)


@pytest.mark.parametrize(
    ("algo_name", "option_name", "option_value"),
    [
        ("fullfact", "n_samples", 3),
        ("SLSQP", "max_iter", 3),
        ("fullfact", "n_samples", None),
        ("SLSQP", "max_iter", None),
    ],
)
def test_set_acquisition_algorithm(
    algo_distribution, input_space, algo_name, option_name, option_value
):
    """Check the setting of the acquisition algorithm used by the MLDataAcquisition."""
    acquisition = MLDataAcquisition(
        "ExpectedImprovement", input_space, algo_distribution
    )
    kwargs = {"algo_name": algo_name}
    options = {}
    if option_value is not None:
        options[option_name] = option_value
        kwargs.update(options)

    acquisition.set_acquisition_algorithm(**kwargs)
    if algo_name == "fullfact":
        algo_options = MLDataAcquisition.default_doe_options.copy()
    else:
        algo_options = MLDataAcquisition.default_opt_options.copy()

    algo_options.update(options)
    assert acquisition._MLDataAcquisition__algo_name == algo_name
    assert acquisition._MLDataAcquisition__algo_options == algo_options


@pytest.mark.parametrize("as_dict", [False, True])
def test_compute(algo_distribution, input_space, as_dict):
    """Check the acquisition of a new point by the MLDataAcquisition."""
    acquisition = MLDataAcquisition(
        "ExpectedImprovement", input_space, algo_distribution
    )
    acquisition.set_acquisition_algorithm("fullfact", n_samples=3)
    x_opt = acquisition.compute_next_input_data(as_dict=as_dict)
    x_opt = x_opt if isinstance(x_opt, ndarray) else x_opt["x"]
    assert x_opt.shape == (1,)


def test_update_algo(algo_distribution_for_update, input_space):
    """Check the update of the machine learning algorithm."""
    distribution = algo_distribution_for_update
    initial_size = len(distribution.learning_set)
    acquisition = MLDataAcquisition("MinimumDistance", input_space, distribution)
    acquisition.set_acquisition_algorithm("fullfact", n_samples=3)
    acquisition.update_algo(AnalyticDiscipline({"y": "1+x"}))
    assert len(distribution.learning_set) == initial_size + 1


@pytest.mark.parametrize(
    ("criterion", "minimize", "options"),
    [("Quantile", True, {"level": 0.1}), ("ExpectedImprovement", False, {})],
)
def test_build_opt_problem_maximize(
    algo_distribution, input_space, criterion, minimize, options
):
    """Check that the optimization problem handles both cost & performance criteria."""
    acquisition = MLDataAcquisition(
        criterion, input_space, algo_distribution, **options
    )
    acquisition._MLDataAcquisition__build_optimization_problem()
    assert acquisition._MLDataAcquisition__problem.minimize_objective == minimize


@pytest.mark.parametrize(
    ("criterion", "use_finite_differences"),
    [("MinimumDistance", True), ("ExpectedImprovement", True)],
)
def test_build_opt_problem_jacobian(
    algo_distribution, input_space, criterion, use_finite_differences
):
    """Check that the optimization problem can use approximated or analytic Jacobian."""
    acquisition = MLDataAcquisition(criterion, input_space, algo_distribution)
    acquisition._MLDataAcquisition__build_optimization_problem()
    assert (
        eq(
            acquisition._MLDataAcquisition__problem.differentiation_method,
            OptimizationProblem.DifferentiationMethod.FINITE_DIFFERENCES,
        )
        == use_finite_differences
    )

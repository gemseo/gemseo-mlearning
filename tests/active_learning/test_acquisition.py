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
from pathlib import Path
from unittest import mock

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.datasets.io_dataset import IODataset
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.algos.gpr import GaussianProcessRegressor
from gemseo.mlearning.regression.algos.linreg import LinearRegressor
from numpy import array
from numpy import ndarray

from gemseo_mlearning.active_learning.acquisition_criteria.minimum.base_minimum import (
    BaseMinimum,
)
from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo
from gemseo_mlearning.active_learning.distributions import KrigingDistribution
from gemseo_mlearning.active_learning.distributions.regressor_distribution import (
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


@pytest.fixture
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
    """Check the initialization of the ActiveLearningAlgo."""
    algo = ActiveLearningAlgo("Minimum", input_space, algo_distribution)
    assert algo.n_initial_samples == 2
    assert algo.regressor_distribution == algo_distribution
    assert algo.regressor == algo_distribution.algo
    assert algo.acquisition_criterion.name == "-EI"
    assert (
        algo._ActiveLearningAlgo__acquisition_algo.algo_name == algo.default_algo_name
    )
    assert (
        algo._ActiveLearningAlgo__acquisition_algo_options == algo.default_opt_options
    )
    assert (
        algo._ActiveLearningAlgo__acquisition_problem.design_space.variable_names
        == ["x"]
    )
    assert algo._ActiveLearningAlgo__acquisition_criterion_family_name == "Minimum"
    assert algo._ActiveLearningAlgo__input_space == input_space
    assert algo._ActiveLearningAlgo__distribution == algo_distribution
    assert algo._ActiveLearningAlgo__acquisition_algo.algo_name == "NLOPT_COBYLA"


def test_init_with_bad_output_dimension(input_space):
    """Check ActiveLearningAlgo initialization raising errors.

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
        NotImplementedError, match="ActiveLearningAlgo works only with scalar output."
    ):
        ActiveLearningAlgo("Minimum", input_space, algo_distribution)


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
    """Check the setting of the acquisition algorithm used by the ActiveLearningAlgo."""
    algo = ActiveLearningAlgo("Minimum", input_space, algo_distribution)
    kwargs = {"algo_name": algo_name}
    options = {}
    if option_value is not None:
        options[option_name] = option_value
        kwargs.update(options)

    algo.set_acquisition_algorithm(**kwargs)
    if algo_name == "fullfact":
        algo_options = ActiveLearningAlgo.default_doe_options.copy()
    else:
        algo_options = ActiveLearningAlgo.default_opt_options.copy()

    algo_options.update(options)
    assert algo._ActiveLearningAlgo__acquisition_algo.algo_name == algo_name
    assert algo._ActiveLearningAlgo__acquisition_algo_options == algo_options


@pytest.mark.parametrize("as_dict", [False, True])
def test_compute(algo_distribution, input_space, as_dict):
    """Check the acquisition of a new point by the ActiveLearningAlgo."""
    algo = ActiveLearningAlgo("Minimum", input_space, algo_distribution)
    algo.set_acquisition_algorithm("fullfact", n_samples=3)
    x_opt = algo.find_next_point(as_dict=as_dict)
    x_opt = x_opt if isinstance(x_opt, ndarray) else x_opt["x"]
    assert x_opt.shape == (1,)


def test_update_algo(algo_distribution_for_update, input_space):
    """Check the update of the machine learning algorithm."""
    distribution = algo_distribution_for_update
    initial_size = len(distribution.learning_set)
    algo = ActiveLearningAlgo(
        "Exploration", input_space, distribution, criterion_name="Distance"
    )
    algo.set_acquisition_algorithm("fullfact", n_samples=3)
    algo.acquire_new_points(AnalyticDiscipline({"y": "1+x"}))
    assert len(distribution.learning_set) == initial_size + 1


def test_plot(algo_distribution_for_update, tmp_wd):
    """Check that a plot is generated when acquiring new points."""
    dataset = IODataset()
    dataset.add_variable(
        "x1", array([[0.0], [1.0], [2.0]]), group_name=dataset.INPUT_GROUP
    )
    dataset.add_variable(
        "x2", array([[1.0], [2.0], [3.0]]), group_name=dataset.INPUT_GROUP
    )
    dataset.add_variable(
        "y",
        array([[1.0], [2.0], [3.0]]),
        group_name=dataset.OUTPUT_GROUP,
    )
    regressor = GaussianProcessRegressor(dataset)
    space = DesignSpace()
    space.add_variable("x1", l_b=0.0, u_b=1.0, value=0.5)
    space.add_variable("x2", l_b=0.0, u_b=1.0, value=0.5)
    algo = ActiveLearningAlgo("Minimum", space, regressor)
    file_path = Path("foo.png")
    algo.acquire_new_points(AnalyticDiscipline({"y": "1+x1+x2"}), file_path=file_path)
    assert file_path.exists()


@pytest.mark.parametrize(
    ("criterion", "minimize"),
    [("Quantile", True), ("Minimum", False)],
)
def test_build_opt_problem_maximize(
    algo_distribution, input_space, criterion, minimize
):
    """Check that the optimization problem handles both cost & performance criteria."""
    options = {}
    if criterion == "Quantile":
        options["level"] = 0.1
        uncertain_space = ParameterSpace()
        uncertain_space.add_random_variable("x", "OTNormalDistribution")
        options["uncertain_space"] = uncertain_space

    algo = ActiveLearningAlgo(criterion, input_space, algo_distribution, **options)
    algo._ActiveLearningAlgo__create_acquisition_problem()
    assert algo._ActiveLearningAlgo__acquisition_problem.minimize_objective == minimize


@pytest.mark.parametrize(
    ("criterion", "kwargs", "use_finite_differences"),
    [("Maximum", {}, True), ("Minimum", {}, True)],
)
def test_build_opt_problem_jacobian(
    algo_distribution, input_space, criterion, use_finite_differences, kwargs
):
    """Check that the optimization problem can use approximated or analytic Jacobian."""
    algo = ActiveLearningAlgo(criterion, input_space, algo_distribution, **kwargs)
    algo._ActiveLearningAlgo__create_acquisition_problem()
    assert (
        eq(
            algo._ActiveLearningAlgo__acquisition_problem.differentiation_method,
            OptimizationProblem.DifferentiationMethod.FINITE_DIFFERENCES,
        )
        == use_finite_differences
    )


@pytest.mark.parametrize(
    ("has_jac", "differentiation_method"),
    [
        (False, OptimizationProblem.DifferentiationMethod.FINITE_DIFFERENCES),
        (True, OptimizationProblem.DifferentiationMethod.USER_GRAD),
    ],
)
def test_analytical_jacobian(
    algo_distribution, input_space, has_jac, differentiation_method
):
    """Check that the OptimizationProblem uses the analytic Jacobian when available."""
    with mock.patch.object(BaseMinimum, "has_jac", has_jac):
        algo = ActiveLearningAlgo("Minimum", input_space, algo_distribution)

    assert (
        algo._ActiveLearningAlgo__acquisition_problem.differentiation_method
        == differentiation_method
    )


@pytest.mark.parametrize(
    ("regressor_class", "regressor_distribution_class"),
    [
        (LinearRegressor, RegressorDistribution),
        (GaussianProcessRegressor, KrigingDistribution),
    ],
)
def test_regressor_at_instantiation(
    input_space, dataset, regressor_class, regressor_distribution_class
):
    """Check that ActiveLearningAlgo can be instantiated from a regressor."""
    regressor = regressor_class(dataset)
    algo = ActiveLearningAlgo("Minimum", input_space, regressor)
    distribution = algo._ActiveLearningAlgo__distribution
    assert isinstance(distribution, regressor_distribution_class)
    assert distribution.algo == regressor

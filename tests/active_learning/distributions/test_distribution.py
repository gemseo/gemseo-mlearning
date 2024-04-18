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

import pytest
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.algos.linreg import LinearRegressor
from gemseo.mlearning.regression.algos.rbf import RBFRegressor
from gemseo.utils.testing.helpers import concretize_classes
from numpy import array
from numpy import linspace
from numpy import newaxis

from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (
    BaseRegressorDistribution,
)
from gemseo_mlearning.active_learning.distributions.regressor_distribution import (
    RegressorDistribution,
)


@pytest.fixture(scope="module")
def distribution(dataset) -> BaseRegressorDistribution:
    """The distribution of a linear regression model built from the learning dataset."""
    with concretize_classes(BaseRegressorDistribution):
        return BaseRegressorDistribution(LinearRegressor(dataset))


@pytest.fixture(scope="module")
def distribution_with_variance(dataset) -> BaseRegressorDistribution:
    """The distribution of a linear regression model built from the learning dataset.

    This distribution has a mock implementation of the method compute_variance.
    """
    with concretize_classes(BaseRegressorDistribution):
        distribution = BaseRegressorDistribution(LinearRegressor(dataset))

    distribution.compute_variance = lambda input_data: input_data
    return distribution


@pytest.fixture(scope="module")
def distribution_with_transformers() -> RegressorDistribution:
    """The distribution of an algorithm using variable transformation."""
    dataset = IODataset()
    x = linspace(-1, 1, 10)[:, None]
    dataset.add_variable("x", x, group_name=dataset.INPUT_GROUP)
    dataset.add_variable("y", x**2, group_name=dataset.OUTPUT_GROUP)

    algo = RBFRegressor(dataset, transformer=RBFRegressor.DEFAULT_TRANSFORMER)

    distribution = RegressorDistribution(algo)
    distribution.learn()
    return distribution


def test_init(distribution):
    """Check the attributes 'algo' and '_samples' after initialization.

    Samples must be None (the value is defined by the 'learn' method) and the algorithm
    must be equal to the provided one.
    """
    assert isinstance(distribution.algo, LinearRegressor)
    assert distribution._samples == []


def test_learning_set(distribution):
    """Check the property ``learning_set``."""
    assert isinstance(distribution.learning_set, IODataset)


def test_inputs_names(distribution):
    """Check the property 'input_names'.

    Must be equal to the names of the inputs of the regression algorithm.
    """
    assert distribution.input_names == distribution.algo.input_names


def test_outputs_names(distribution):
    """Check the property 'output_names'.

    Should be equal to the names of the outputs of the regression algorithm.
    """
    assert distribution.output_names == distribution.algo.output_names


def test_output_dimension(distribution):
    """Check the property 'output_dimension'.

    Must be equal to the output dimension of the regression algorithm.
    """
    assert distribution.output_dimension == distribution.algo.output_dimension


@pytest.mark.parametrize(
    ("samples", "expected_samples", "prediction"),
    [(None, range(3), 2.0 / 3), ([0, 2], [0, 2], 1.0)],
)
def test_learn(distribution, samples, expected_samples, prediction):
    """Check the learning stage.

    The algorithm must be trained, the samples must be equal to the provided ones (or to
    all samples if None is provided) and the prediction must be right Should be equal to
    the output dimension of the regression algorithm.
    """
    distribution.learn(samples)
    assert distribution.algo.is_trained
    assert distribution._samples == expected_samples
    assert distribution.algo.predict(array([0.0]))[0] == pytest.approx(prediction, 0.01)


@pytest.mark.parametrize(
    ("input_data", "output_data"),
    [
        (array([0.0]), array([1.0])),
        (array([[0.0]]), array([[1.0]])),
        ({"x": array([0.0])}, {"y": array([1.0])}),
        ({"x": array([[0.0]])}, {"y": array([[1.0]])}),
    ],
)
def test_predict(distribution, input_data, output_data):
    """Check the prediction stage."""
    distribution.learn([0, 2])
    assert distribution.predict(input_data) == output_data


@pytest.mark.parametrize(
    ("input_data", "output_data"),
    [
        (array([4.0]), array([2.0])),
        ({"x": array([4.0])}, {"y": array([2.0])}),
        (array([[4.0]]), array([[2.0]])),
        ({"x": array([[4.0]])}, {"y": array([[2.0]])}),
    ],
)
def test_standard_deviation(distribution_with_variance, input_data, output_data):
    """Check that the standard deviation is the square root of the variance.

    This test method uses a mock distribution whose method compute_variance is the
    identity function. Thus, the compute_standard_deviation method should return the
    square root of its input.
    """
    assert (
        distribution_with_variance.compute_standard_deviation(input_data) == output_data
    )


def test_change_learning_set(dataset):
    """Check that changing the learning set updates the algorithm."""
    with concretize_classes(BaseRegressorDistribution):
        distribution = BaseRegressorDistribution(LinearRegressor(dataset))

    new_dataset = IODataset()
    new_dataset.add_variable(
        "x", array([0.0, 1.0])[:, None], group_name=new_dataset.INPUT_GROUP
    )
    new_dataset.add_variable(
        "y",
        array([1.0, 1.0])[:, None],
        group_name=new_dataset.OUTPUT_GROUP,
    )
    distribution.change_learning_set(new_dataset)
    assert len(distribution.learning_set) == 2
    assert distribution.predict(array([0.5])) == array([1.0])


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("is_2d", [False, True])
@pytest.mark.parametrize(
    "statistic",
    ["variance", "mean", "expected_improvement", "standard_deviation"],
)
def test_decorator_for_statistics(
    distribution_with_transformers, as_dict, is_2d, statistic
):
    """Check the use of decorator for the methods compute_{statistic}."""
    input_data = array([0.0])
    if is_2d:
        input_data = input_data[newaxis, :]

    if as_dict:
        input_data = {"x": input_data}

    args = ()
    if statistic == "expected_improvement":
        args = (0.0,)

    compute_statistic = getattr(distribution_with_transformers, f"compute_{statistic}")
    output_data = compute_statistic(input_data, *args)

    assert isinstance(output_data, dict) == as_dict

    if as_dict:
        output_data = output_data["y"]

    assert output_data.ndim == 1 + is_2d

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
from gemseo.core.dataset import Dataset
from gemseo.mlearning.regression.linreg import LinearRegressor
from gemseo_mlearning.adaptive.distribution import MLRegressorDistribution
from numpy import array


@pytest.fixture(scope="module")
def distribution(dataset) -> MLRegressorDistribution:
    """The distribution of a linear regression model built from the learning dataset."""
    return MLRegressorDistribution(LinearRegressor(dataset))


@pytest.fixture(scope="module")
def distribution_with_variance(dataset) -> MLRegressorDistribution:
    """The distribution of a linear regression model built from the learning dataset.

    This distribution has a mock implementation of the method compute_variance.
    """
    distribution = MLRegressorDistribution(LinearRegressor(dataset))
    distribution.compute_variance = lambda input_data: input_data
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
    assert isinstance(distribution.learning_set, Dataset)


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
    "samples,expected_samples,prediction",
    [(None, range(0, 3), 2.0 / 3), ([0, 2], [0, 2], 1.0)],
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
    "input_data,output_data",
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
    "input_data,output_data",
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


def test_notimplementederror(distribution):
    """Check that calling non implemented methods raises errors."""
    with pytest.raises(NotImplementedError):
        distribution.compute_confidence_interval(array([0.0]))

    with pytest.raises(NotImplementedError):
        distribution.compute_mean(array([0.0]))

    with pytest.raises(NotImplementedError):
        distribution.compute_variance(array([0.0]))

    with pytest.raises(NotImplementedError):
        distribution.compute_expected_improvement(array([0.0]), 0.0)


def test_change_learning_set(dataset):
    """Check that changing the learning set updates the algorithm."""
    distribution = MLRegressorDistribution(LinearRegressor(dataset))
    new_dataset = Dataset()
    new_dataset.add_variable(
        "x", array([0.0, 1.0])[:, None], group=new_dataset.INPUT_GROUP
    )
    new_dataset.add_variable(
        "y",
        array([1.0, 1.0])[:, None],
        group=new_dataset.OUTPUT_GROUP,
        cache_as_input=False,
    )
    distribution.change_learning_set(new_dataset)
    assert len(distribution.learning_set) == 2
    assert distribution.predict(array([0.5])) == array([1.0])

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
from gemseo.mlearning.regression.linreg import LinearRegressor
from gemseo.utils.testing.helpers import concretize_classes
from numpy import array
from numpy import quantile

from gemseo_mlearning.adaptive.criteria.distances.criterion_min import MinimumDistance
from gemseo_mlearning.adaptive.criteria.expectation.criterion import Expectation
from gemseo_mlearning.adaptive.criteria.mean_std.criterion import MeanSigma
from gemseo_mlearning.adaptive.criteria.optimum.criterion import ExpectedImprovement
from gemseo_mlearning.adaptive.criteria.optimum.criterion_max import (
    MaxExpectedImprovement,
)
from gemseo_mlearning.adaptive.criteria.optimum.criterion_min import (
    MinExpectedImprovement,
)
from gemseo_mlearning.adaptive.criteria.quantile.criterion import Quantile
from gemseo_mlearning.adaptive.criteria.standard_deviation.criterion import (
    StandardDeviation,
)
from gemseo_mlearning.adaptive.criteria.value.criterion import LimitState
from gemseo_mlearning.adaptive.criteria.variance.criterion import Variance
from gemseo_mlearning.adaptive.distribution import MLRegressorDistribution


@pytest.fixture(scope="module")
def algo_distribution() -> MLRegressorDistribution:
    """A mock distribution of a regression model.

    This distribution uses mocks for the methods compute_variance and compute_mean.
    """
    dataset = IODataset()
    dataset.add_variable(
        "x", array([0.0, 0.5, 1.0])[:, None], group_name=dataset.INPUT_GROUP
    )
    dataset.add_variable(
        "y",
        array([1.0, 0.0, 1.0])[:, None],
        group_name=dataset.OUTPUT_GROUP,
    )
    with concretize_classes(MLRegressorDistribution):
        distribution = MLRegressorDistribution(LinearRegressor(dataset))

    distribution.learn()
    distribution.compute_variance = lambda input_data: 2 * input_data
    distribution.compute_mean = lambda input_data: 3 * input_data

    def expected_improvement(input_data, minimum, maximize=True):
        return 4 * input_data

    distribution.compute_expected_improvement = expected_improvement
    return distribution


def test_variance(algo_distribution):
    """Check the criterion Variance."""
    criterion = Variance(algo_distribution)
    result = criterion(array([0.0]))
    assert result.shape == (1,)
    expected_variance = algo_distribution.compute_variance(result)
    assert result * criterion.output_range == pytest.approx(expected_variance, abs=0.1)


def test_standard_deviation(algo_distribution):
    """Check the criterion StandardDeviation."""
    value = array([0.0])
    expected_std = algo_distribution.compute_standard_deviation(value)
    criterion = StandardDeviation(algo_distribution)
    assert criterion(value) * criterion.output_range == expected_std


def test_expected_improvement(algo_distribution):
    """Check the criterion ExpectedImprovement."""
    value = array([0.0])
    minimum = algo_distribution.learning_set.get_view(variable_names="y").min()
    criterion = ExpectedImprovement(algo_distribution)
    expected = algo_distribution.compute_expected_improvement(value, minimum)
    assert criterion(value) * criterion.output_range == expected


def test_expected_improvement_for_minimum(algo_distribution):
    """Check the criterion MinExpectedImprovement."""
    value = array([0.0])
    minimum = algo_distribution.learning_set.get_view(variable_names="y").min()
    criterion = MinExpectedImprovement(algo_distribution)
    expected = algo_distribution.compute_expected_improvement(value, minimum)
    assert criterion(value) * criterion.output_range == expected


def test_expected_improvement_for_maximum(algo_distribution):
    """Check the criterion MinExpectedImprovement."""
    value = array([0.0])
    criterion = MaxExpectedImprovement(algo_distribution)
    maximum = algo_distribution.learning_set.get_view(variable_names="y").max()
    expected = algo_distribution.compute_expected_improvement(value, maximum, True)
    assert criterion(value) * criterion.output_range == expected


def test_expectation(algo_distribution):
    """Check the criterion Expectation."""
    value = array([0.5])
    criterion = Expectation(algo_distribution)
    expected = algo_distribution.compute_mean(value)
    assert criterion(value) * criterion.output_range == expected


def test_mean_sigma(algo_distribution):
    """Check the criterion MeanSigma."""
    value = array([0.5])
    criterion = MeanSigma(algo_distribution, 2.0)
    expected = algo_distribution.compute_mean(
        value
    ) + 2.0 * algo_distribution.compute_standard_deviation(value)
    assert criterion(value) * criterion.output_range == expected


@pytest.mark.parametrize(("value", "expected"), [(0.5, 0.0), (0.25, 1.0), (0.125, 0.5)])
def test_minimum_distance(algo_distribution, value, expected):
    """Check the criterion MinimumDistance."""
    criterion = MinimumDistance(algo_distribution)
    value = criterion(array([value]))
    assert value == array([expected])


def test_limitstate(algo_distribution):
    """Check the criterion LimitState."""
    limit_state = 0.5
    value = array([0.25])
    mean = algo_distribution.compute_mean(value)
    std = algo_distribution.compute_standard_deviation(value)
    expected = abs(limit_state - mean) / std
    criterion = LimitState(algo_distribution, limit_state)
    assert criterion(value) == expected


def test_quantile(algo_distribution):
    """Check the criterion Quantile."""
    level = 0.8
    quantile_ = quantile(
        algo_distribution.learning_set.get_view(variable_names="y"), level
    )
    value = array([0.25])
    mean = algo_distribution.compute_mean(value)
    std = algo_distribution.compute_standard_deviation(value)
    expected = abs(quantile_ - mean) / std
    criterion = Quantile(algo_distribution, level)
    assert criterion(value) == expected

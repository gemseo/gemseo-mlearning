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

from operator import add
from operator import mul
from operator import sub
from operator import truediv

import pytest
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.mlearning.regression.linreg import LinearRegressor
from gemseo_mlearning.adaptive.criteria.distances.criterion_min import (
    MinimumDistance,
)
from gemseo_mlearning.adaptive.criteria.optimum.criterion import (
    ExpectedImprovement,
)
from gemseo_mlearning.adaptive.criterion import (
    MLDataAcquisitionCriterionFactory,
)
from gemseo_mlearning.adaptive.distributions.regressor_distribution import (
    RegressorDistribution,
)
from numpy import array


@pytest.fixture(scope="module")
def algo_distribution(dataset) -> RegressorDistribution:
    """The distribution of the linear regression model."""
    distribution = RegressorDistribution(LinearRegressor(dataset), size=3)
    distribution.learn()
    return distribution


def test_factory(algo_distribution):
    """Check the MLDataAcquisitionCriterionFactory."""
    factory = MLDataAcquisitionCriterionFactory()
    criterion = factory.create("ExpectedImprovement", algo_distribution)
    assert isinstance(criterion, ExpectedImprovement)
    assert "ExpectedImprovement" in factory.available_criteria
    assert factory.is_available("ExpectedImprovement")


@pytest.fixture(scope="module")
def criterion_1(algo_distribution):
    """First data acquisition criterion to check operations."""
    return ExpectedImprovement(algo_distribution)


@pytest.fixture(scope="module")
def criterion_2(algo_distribution):
    """Second data acquisition criterion to check operations."""
    return MinimumDistance(algo_distribution)


@pytest.mark.parametrize("operator", [add, mul, sub, truediv])
@pytest.mark.parametrize("second_operand", [2.0, None])
def test_operations(criterion_1, criterion_2, operator, second_operand):
    """Check elementary operations with two MLDataAcquisitionCriterion."""
    x = array([0.5])
    c_1 = criterion_1(x)
    if second_operand is None:
        c_2 = criterion_2(x)
        criterion_2 = criterion_2
    else:
        c_2 = second_operand
        criterion_2 = second_operand

    criterion_3 = operator(criterion_1, criterion_2)
    assert id(criterion_3.algo_distribution) == id(criterion_1.algo_distribution)
    assert id(criterion_3.output_range) == id(criterion_1.output_range)
    assert criterion_3(x) == operator(c_1, c_2)


def test_neg(criterion_1):
    """Check the neg operator."""
    x = array([0.5])
    neg_criterion_1 = -criterion_1
    assert neg_criterion_1(x) == -criterion_1(x)


def test_linear_combination(algo_distribution):
    """Check that a combination of MLDataAcquisitionCriterion is an MDOFunction."""
    criterion_1 = ExpectedImprovement(algo_distribution)
    criterion_2 = MinimumDistance(algo_distribution)
    criterion_3 = criterion_1 * 0.2 + criterion_2 * 0.8
    assert isinstance(criterion_3, MDOFunction)
    x_new = array([0.5])
    assert criterion_3(x_new) == criterion_1(x_new) * 0.2 + criterion_2(x_new) * 0.8


def test_scaling_factor(dataset, algo_distribution):
    """Check that the scaling factor is updated with the output range."""
    criterion = ExpectedImprovement(algo_distribution)
    assert criterion._scaling_factor == 1.0
    criterion.output_range = 2.0
    assert criterion._scaling_factor == 2.0
    criterion.output_range = 0.0
    assert criterion._scaling_factor == 1.0

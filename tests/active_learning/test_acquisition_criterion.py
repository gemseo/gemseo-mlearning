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
from gemseo.mlearning.regression.algos.linreg import LinearRegressor
from numpy import array

from gemseo_mlearning.active_learning.acquisition_criteria.exploration.distance import (
    Distance,
)
from gemseo_mlearning.active_learning.acquisition_criteria.minimum.ei import EI
from gemseo_mlearning.active_learning.distributions.regressor_distribution import (
    RegressorDistribution,
)


@pytest.fixture(scope="module")
def algo_distribution(dataset) -> RegressorDistribution:
    """The distribution of the linear regression model."""
    distribution = RegressorDistribution(LinearRegressor(dataset), size=3)
    distribution.learn()
    return distribution


@pytest.fixture(scope="module")
def criterion_1(algo_distribution):
    """First data acquisition criterion to check operations."""
    return EI(algo_distribution)


@pytest.fixture(scope="module")
def criterion_2(algo_distribution):
    """Second data acquisition criterion to check operations."""
    return Distance(algo_distribution)


@pytest.mark.parametrize("operator", [add, mul, sub, truediv])
@pytest.mark.parametrize("second_operand", [2.0, None])
def test_operations(criterion_1, criterion_2, operator, second_operand):
    """Check elementary operations with two MLDataAcquisitionCriterion."""
    x = array([0.5])
    c_1 = criterion_1.func(x)
    if second_operand is None:
        c_2 = criterion_2.func(x)
        criterion_2 = criterion_2
    else:
        c_2 = second_operand
        criterion_2 = second_operand

    criterion_3 = operator(criterion_1, criterion_2)
    assert criterion_3.func(x) == operator(c_1, c_2)


def test_neg(criterion_1):
    """Check the neg operator."""
    x = array([0.5])
    neg_criterion_1 = -criterion_1
    assert neg_criterion_1.func(x) == -criterion_1.func(x)


def test_linear_combination(algo_distribution):
    """Check that a combination of MLDataAcquisitionCriterion is an MDOFunction."""
    criterion_1 = EI(algo_distribution)
    criterion_2 = Distance(algo_distribution)
    criterion_3 = criterion_1 * 0.2 + criterion_2 * 0.8
    assert isinstance(criterion_3, MDOFunction)
    x_new = array([0.5])
    assert (
        criterion_3.func(x_new)
        == criterion_1.func(x_new) * 0.2 + criterion_2.func(x_new) * 0.8
    )

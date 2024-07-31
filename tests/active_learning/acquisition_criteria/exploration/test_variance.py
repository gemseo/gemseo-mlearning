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
from numpy import array
from numpy.testing import assert_equal

from gemseo_mlearning.active_learning.acquisition_criteria.exploration.variance import (  # noqa: E501
    Variance,
)


@pytest.mark.parametrize(
    ("input_value", "shape"), [(array([0.0]), (1,)), (array([[0.0], [0.0]]), (2, 1))]
)
def test_variance(algo_distribution, input_value, shape):
    """Check the Variance criterion."""
    variance = algo_distribution.compute_variance(input_value)
    criterion = Variance(algo_distribution)
    output_range = criterion.output_range
    assert variance.shape == shape
    assert_equal(criterion.evaluate(input_value), variance / output_range**2)

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

from gemseo_mlearning.active_learning.acquisition_criteria.maximum.mean import (  # noqa: E501
    Mean,
)


@pytest.mark.parametrize(
    ("input_value", "shape"), [(array([0.5]), (1,)), (array([[0.5], [0.5]]), (2, 1))]
)
def test_mean(algo_distribution, input_value, shape):
    """Check the Mean criterion."""
    criterion = Mean(algo_distribution)
    mean = algo_distribution.compute_mean(input_value)
    output_range = criterion.output_range
    assert mean.shape == shape
    assert_equal(criterion.evaluate(input_value), mean / output_range)

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

from gemseo_mlearning.active_learning.acquisition_criteria.minimum.lcb import LCB


@pytest.mark.parametrize(("kwargs", "kappa"), [({}, 2.0), ({"kappa": 3.0}, 3.0)])
@pytest.mark.parametrize(
    ("input_value", "shape"), [(array([0.5]), (1,)), (array([[0.5], [0.5]]), (2, 1))]
)
def test_lcb(algo_distribution, kwargs, kappa, input_value, shape):
    """Check the LCB criterion."""
    criterion = LCB(algo_distribution, **kwargs)
    mean = algo_distribution.compute_mean(input_value)
    std = algo_distribution.compute_standard_deviation(input_value)
    output_range = criterion.output_range
    assert mean.shape == std.shape == shape
    assert_equal(criterion.evaluate(input_value), (mean - kappa * std) / output_range)

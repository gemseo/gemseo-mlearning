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
from numpy.testing import assert_almost_equal

from gemseo_mlearning.active_learning.acquisition_criteria.maximum.ei import EI


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [(array([0.5]), array([0.0])), (array([[0.5], [0.5]]), array([[0.0], [0.0]]))],
)
def test_ei(algo_distribution, input_value, expected):
    """Check the EI criterion."""
    criterion = EI(algo_distribution)
    assert_almost_equal(criterion.evaluate(input_value), expected)

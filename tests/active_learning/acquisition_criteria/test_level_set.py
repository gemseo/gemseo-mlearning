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
from numpy import inf
from numpy import nan_to_num
from numpy.testing import assert_almost_equal

from gemseo_mlearning.active_learning.acquisition_criteria.level_set.ef import EF
from gemseo_mlearning.active_learning.acquisition_criteria.level_set.ei import EI
from gemseo_mlearning.active_learning.acquisition_criteria.level_set.u import U


@pytest.mark.parametrize(
    ("cls", "input_value", "expected"),
    [
        (U, array([0.25]), array([0.35355339])),
        (EF, array([0.25]), array([0.2472296])),
        (EI, array([0.25]), array([0.2298589])),
        (U, array([[0.25]] * 2), array([[0.35355339]] * 2)),
        (EF, array([[0.25]] * 2), array([[0.2472296]] * 2)),
        (EI, array([[0.25]] * 2), array([[0.2298589]] * 2)),
    ],
)
def test_level_set(algo_distribution, cls, input_value, expected):
    """Check the criteria deriving from BaseLevelSet."""
    criterion = cls(algo_distribution, 0.5)
    assert_almost_equal(criterion.func(input_value), expected)


def test_u_at_training_point(algo_distribution):
    """Check that the U criterion at a training point is infinity."""
    u = U(algo_distribution, 1)
    u._compute_standard_deviation = lambda x: array([0])
    infinity = nan_to_num(inf)

    # Case where mean=mu(x) and std=0 => U = abs(1-mu(x))/0
    assert_almost_equal(u.evaluate(array([0.0])), array([infinity]))

    # Case where mean=1 and std=0 => U = abs(1-1)/0 = 0/0
    u._compute_mean = lambda x: array([1])
    assert_almost_equal(u.evaluate(array([0.0])), array([infinity]))

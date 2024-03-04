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

from numpy import array

from gemseo_mlearning.active_learning.acquisition_criteria.limit_state import LimitState


def test_limit_state(algo_distribution):
    """Check the criterion LimitState."""
    limit_state = 0.5
    value = array([0.25])
    mean = algo_distribution.compute_mean(value)
    std = algo_distribution.compute_standard_deviation(value)
    expected = abs(limit_state - mean) / std
    criterion = LimitState(algo_distribution, limit_state)
    assert criterion(value) == expected

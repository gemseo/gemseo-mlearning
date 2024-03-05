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

from gemseo_mlearning.active_learning.acquisition_criteria.level_set import LevelSet


def test_level_set(algo_distribution):
    """Check the criterion LevelSet."""
    output_target = 0.5
    input_value = array([0.25])
    mean = algo_distribution.compute_mean(input_value)
    std = algo_distribution.compute_standard_deviation(input_value)
    expected = abs(output_target - mean) / std
    criterion = LevelSet(algo_distribution, output_target)
    assert criterion(input_value) == expected

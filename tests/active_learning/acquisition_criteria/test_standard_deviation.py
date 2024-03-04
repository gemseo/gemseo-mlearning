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

from gemseo_mlearning.active_learning.acquisition_criteria.standard_deviation import (  # noqa: E501
    StandardDeviation,
)


def test_standard_deviation(algo_distribution):
    """Check the criterion StandardDeviation."""
    value = array([0.0])
    expected_std = algo_distribution.compute_standard_deviation(value)
    criterion = StandardDeviation(algo_distribution)
    assert criterion(value) * criterion.output_range == expected_std

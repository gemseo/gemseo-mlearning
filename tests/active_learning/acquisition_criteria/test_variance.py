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

from gemseo_mlearning.active_learning.acquisition_criteria.variance import Variance


def test_variance(algo_distribution):
    """Check the criterion Variance."""
    criterion = Variance(algo_distribution)
    result = criterion(array([0.0]))
    assert result.shape == (1,)
    expected_variance = algo_distribution.compute_variance(result)
    assert result * criterion.output_range == pytest.approx(expected_variance, abs=0.1)

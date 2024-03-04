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
from numpy import quantile

from gemseo_mlearning.active_learning.acquisition_criteria.quantile import Quantile


def test_quantile(algo_distribution):
    """Check the criterion Quantile."""
    level = 0.8
    quantile_ = quantile(
        algo_distribution.learning_set.get_view(variable_names="y"), level
    )
    value = array([0.25])
    mean = algo_distribution.compute_mean(value)
    std = algo_distribution.compute_standard_deviation(value)
    expected = abs(quantile_ - mean) / std
    criterion = Quantile(algo_distribution, level)
    assert criterion(value) == expected

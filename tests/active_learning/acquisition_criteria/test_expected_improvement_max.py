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

from gemseo_mlearning.active_learning.acquisition_criteria.max_expected_improvement import (  # noqa: E501
    MaxExpectedImprovement,
)


def test_expected_improvement_for_maximum(algo_distribution):
    """Check the criterion MinExpectedImprovement."""
    value = array([0.0])
    criterion = MaxExpectedImprovement(algo_distribution)
    maximum = algo_distribution.learning_set.get_view(variable_names="y").max()
    expected = algo_distribution.compute_expected_improvement(value, maximum, True)
    assert criterion(value) * criterion.output_range == expected

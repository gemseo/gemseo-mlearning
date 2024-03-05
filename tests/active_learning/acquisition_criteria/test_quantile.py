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

import re

import pytest
from gemseo.algos.parameter_space import ParameterSpace
from numpy import array
from numpy.testing import assert_almost_equal

from gemseo_mlearning.active_learning.acquisition_criteria.quantile import Quantile


def test_quantile(algo_distribution):
    """Check the criterion Quantile."""
    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable("a", "OTNormalDistribution")
    uncertain_space.add_random_variable("x", "OTNormalDistribution")
    criterion = Quantile(algo_distribution, 0.8, uncertain_space)
    assert_almost_equal(criterion(array([0.25])), array([0.11785113]))


def test_quantile_error(algo_distribution):
    """Check the exception raised by the criterion Quantile."""
    uncertain_space = ParameterSpace()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The probability distributions of the input variables x are missing."
        ),
    ):
        Quantile(algo_distribution, 0.8, uncertain_space)

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
"""Tests for the getter of regression distributions."""
from __future__ import annotations

import pytest

from gemseo_mlearning.adaptive.distributions import KrigingDistribution
from gemseo_mlearning.adaptive.distributions import RegressorDistribution
from gemseo_mlearning.adaptive.distributions import get_regressor_distribution


@pytest.mark.parametrize(
    ("algorithm", "class_"),
    [("linear_algo", RegressorDistribution), ("kriging_algo", KrigingDistribution)],
)
def test_get_regressor_distribution(request, algorithm, class_):
    """Check the getter of the distribution of a regression algorithm."""
    assert isinstance(
        get_regressor_distribution(request.getfixturevalue(algorithm)), class_
    )

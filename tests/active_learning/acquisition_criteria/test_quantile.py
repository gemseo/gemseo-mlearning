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

from gemseo_mlearning.active_learning.acquisition_criteria.quantile.ef import EF
from gemseo_mlearning.active_learning.acquisition_criteria.quantile.ei import EI
from gemseo_mlearning.active_learning.acquisition_criteria.quantile.u import U


@pytest.fixture(scope="module")
def uncertain_space() -> ParameterSpace:
    """An uncertain space."""
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable("a", "OTNormalDistribution")
    parameter_space.add_random_variable("x", "OTNormalDistribution")
    return parameter_space


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (U, 0.1178511),
        (EF, 0.259206),
        (EI, 0.2405944),
    ],
)
def test_quantile_variants(algo_distribution, uncertain_space, cls, expected):
    """Check the criteria deriving from BaseQuantile with a Kriging distribution."""
    criterion = cls(algo_distribution, 0.8, uncertain_space)
    assert_almost_equal(criterion(array([0.25])), array([expected]))


def test_quantile_error(algo_distribution):
    """Check the exception raised by a BaseQuantile criterion."""
    uncertain_space = ParameterSpace()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The probability distributions of the input variables x are missing."
        ),
    ):
        EF(algo_distribution, 0.8, uncertain_space)

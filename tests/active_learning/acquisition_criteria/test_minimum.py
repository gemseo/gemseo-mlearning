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
from numpy.testing import assert_almost_equal

from gemseo_mlearning.active_learning.acquisition_criteria.minimum.ei import EI
from gemseo_mlearning.active_learning.acquisition_criteria.minimum.lcb import LCB
from gemseo_mlearning.active_learning.acquisition_criteria.minimum.output import Output


def test_minimum_kriging(kriging_distribution):
    """Check the criterion EI with a Kriging distribution."""
    criterion = EI(kriging_distribution)
    assert_almost_equal(criterion.func(array([0.0])), array([0.0]))


@pytest.mark.parametrize(
    ("cls", "options", "expected"),
    [
        (EI, {}, 0.0),
        (LCB, {}, 1.0),
        (LCB, {"kappa": 3.0}, 1.0),
        (Output, {}, 1.0),
    ],
)
def test_minimum_kriging_regressor(kriging_distribution, cls, options, expected):
    """Check the criteria deriving from BaseMinimum with a Kriging distribution."""
    criterion = cls(kriging_distribution, **options)
    assert_almost_equal(criterion.func(array([0.0])), array([expected]))


@pytest.mark.parametrize(
    ("cls", "options", "expected"),
    [
        (EI, {}, [0.75]),
        (LCB, {}, -0.62),
        (LCB, {"kappa": 3.0}, -1.12),
        (Output, {}, 0.37),
    ],
)
def test_minimum_regressor(algo_distribution, cls, options, expected):
    """Check the criteria deriving from BaseMinimum with a non-Kriging distribution."""
    criterion = cls(algo_distribution, **options)
    assert_almost_equal(criterion.func(array([0.123])), array([expected]), decimal=2)

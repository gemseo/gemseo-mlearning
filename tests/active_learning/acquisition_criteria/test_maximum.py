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

from gemseo_mlearning.active_learning.acquisition_criteria.maximum.ei import EI
from gemseo_mlearning.active_learning.acquisition_criteria.maximum.output import Output
from gemseo_mlearning.active_learning.acquisition_criteria.maximum.ucb import UCB


@pytest.mark.parametrize(
    ("cls", "options", "expected"),
    [
        (EI, {}, 0.0),
        (UCB, {}, 1.0),
        (UCB, {"kappa": 3.0}, 1.0),
        (Output, {}, 1.0),
    ],
)
def test_maximum_kriging_regressor(kriging_distribution, cls, options, expected):
    """Check the criteria deriving from BaseMaximum with a Kriging distribution."""
    criterion = cls(kriging_distribution, **options)
    assert_almost_equal(criterion.func(array([0.0])), array([expected]))


@pytest.mark.parametrize(
    ("cls", "options", "expected"),
    [
        (EI, {}, [0.0]),
        (UCB, {}, 1.3609677),
        (UCB, {"kappa": 3.0}, 1.8569516),
        (Output, {}, 0.369),
    ],
)
def test_maximum_regressor(algo_distribution, options, cls, expected):
    """Check the criteria deriving from BaseMaximum with a RegressorDistribution."""
    criterion = cls(algo_distribution, **options)
    assert_almost_equal(criterion.func(array([0.123])), array([expected]))

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
from numpy import array
from numpy.testing import assert_almost_equal

from gemseo_mlearning.active_learning.acquisition_criteria.minimum.ei import EI
from gemseo_mlearning.active_learning.acquisition_criteria.minimum.lcb import LCB
from gemseo_mlearning.active_learning.acquisition_criteria.minimum.output import Output


@pytest.mark.parametrize(
    ("cls", "options", "input_value", "expected"),
    [
        (EI, {}, array([0.0]), array([0.0])),
        (LCB, {}, array([0.0]), array([1.0])),
        (LCB, {"kappa": 3.0}, array([0.0]), array([1.0])),
        (Output, {}, array([0.0]), array([1.0])),
        (EI, {}, array([[0.0]] * 2), array([[0.0]] * 2)),
        (LCB, {}, array([[0.0]] * 2), array([[1.0]] * 2)),
        (LCB, {"kappa": 3.0}, array([[0.0]] * 2), array([[1.0]] * 2)),
        (Output, {}, array([[0.0]] * 2), array([[1.0]] * 2)),
    ],
)
def test_minimum_kriging_regressor(
    kriging_distribution, cls, options, input_value, expected
):
    """Check the criteria deriving from BaseMinimum with a Kriging distribution."""
    criterion = cls(kriging_distribution, **options)
    assert_almost_equal(criterion.func(input_value), expected)


@pytest.mark.parametrize(
    ("cls", "options", "input_value", "expected"),
    [
        (EI, {}, array([0.123]), array([0.75])),
        (LCB, {}, array([0.123]), array([-0.620])),
        (LCB, {"kappa": 3.0}, array([0.123]), array([-1.12])),
        (Output, {}, array([0.123]), array([0.37])),
        (EI, {}, array([[0.123]] * 2), array([[0.75]] * 2)),
        (LCB, {}, array([[0.123]] * 2), array([[-0.62]] * 2)),
        (LCB, {"kappa": 3.0}, array([[0.123]] * 2), array([[-1.12]] * 2)),
        (Output, {}, array([[0.123]] * 2), array([[0.37]] * 2)),
    ],
)
def test_minimum_regressor(algo_distribution, cls, options, input_value, expected):
    """Check the criteria deriving from BaseMinimum with a non-Kriging distribution."""
    criterion = cls(algo_distribution, **options)
    assert_almost_equal(criterion.func(input_value), expected, decimal=2)


@pytest.mark.parametrize(
    ("cls", "input_value"),
    [
        (EI, array([0.123])),
        (LCB, array([0.123])),
        (Output, array([0.123])),
    ],
)
def test_bad_parallel_regressor(algo_distribution, cls, input_value):
    """Check that parallelized criteria with non GP regressor lead to failure."""
    criterion = cls(algo_distribution, batch_size=2)

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "Parallelization with batch_size > 1 is not yet implemented "
            "for regressors that are not based on a random process."
        ),
    ):
        criterion.func(input_value)

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
    ("cls", "input_value", "expected"),
    [
        (U, array([0.25]), array([8.5598195e-14])),
        (EF, array([0.25]), array([0.5746878])),
        (EI, array([0.25]), array([0.6843251])),
        (U, array([[0.25]] * 2), array([[8.5598195e-14]] * 2)),
        (EF, array([[0.25]] * 2), array([[0.5746878]] * 2)),
        (EI, array([[0.25]] * 2), array([[0.6843251]] * 2)),
    ],
)
def test_quantile_kriging(
    kriging_distribution, uncertain_space, cls, input_value, expected
):
    """Check the criteria deriving from BaseQuantile with a Kriging distribution."""
    criterion = cls(kriging_distribution, 0.8, uncertain_space)
    assert_almost_equal(criterion.func(input_value), expected)


@pytest.mark.parametrize(
    ("cls", "input_value", "expected"),
    [
        (U, array([0.25]), array([0.11785113])),
        (EF, array([0.25]), array([0.25])),
        (EI, array([0.25]), array([0.64])),
        (U, array([[0.25]] * 2), array([[0.11785113]] * 2)),
        (EF, array([[0.25]] * 2), array([[0.25]] * 2)),
        (EI, array([[0.25]] * 2), array([[0.64]] * 2)),
    ],
)
def test_quantile_regressor(
    algo_distribution, cls, input_value, expected, uncertain_space
):
    """Check the criteria deriving from BaseQuantile with non GP regressor."""
    criterion = cls(
        regressor_distribution=algo_distribution,
        level=0.8,
        uncertain_space=uncertain_space,
    )
    assert_almost_equal(criterion.func(input_value), expected, decimal=2)


@pytest.mark.parametrize(
    ("cls", "input_value", "expected"),
    [
        (U, array([[0.25]] * 2), array([1.0013774e-08])),
        (EF, array([[0.25]] * 2), array([0.9427619])),
        (EI, array([[0.25]] * 2), array([0.8888889])),
    ],
)
def test_quantile_parallel(
    kriging_distribution, cls, input_value, expected, uncertain_space
):
    """Check the parallelized criteria deriving from BaseQuantile."""
    criterion = cls(
        regressor_distribution=kriging_distribution,
        level=0.8,
        uncertain_space=uncertain_space,
        batch_size=2,
    )
    assert_almost_equal(criterion.func(input_value), expected)


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


@pytest.mark.parametrize(
    "cls",
    [
        U,
        EI,
        EF,
    ],
)
def test_bad_parallel_regressor(algo_distribution, cls, uncertain_space):
    """Check that parallelized criteria with non GP regressor lead to failure."""
    criterion = cls(
        regressor_distribution=algo_distribution,
        level=0.8,
        uncertain_space=uncertain_space,
        batch_size=2,
    )

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "Parallelization with batch_size > 1 is not yet implemented "
            "for regressors that are not based on a random process."
        ),
    ):
        criterion.func(array([0.123]))

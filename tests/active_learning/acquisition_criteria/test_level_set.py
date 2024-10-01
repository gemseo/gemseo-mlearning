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
from numpy import inf
from numpy import nan_to_num
from numpy.testing import assert_almost_equal

from gemseo_mlearning.active_learning.acquisition_criteria.level_set.ef import EF
from gemseo_mlearning.active_learning.acquisition_criteria.level_set.ei import EI
from gemseo_mlearning.active_learning.acquisition_criteria.level_set.u import U


@pytest.mark.parametrize(
    ("cls", "input_value", "expected"),
    [
        (U, array([0.25]), array([0.3535534])),
        (EF, array([0.25]), array([0.5547019])),
        (EI, array([0.25]), array([0.6640591])),
        (U, array([[0.25]] * 2), array([[0.3535534]] * 2)),
        (EF, array([[0.25]] * 2), array([[0.5547019]] * 2)),
        (EI, array([[0.25]] * 2), array([[0.6640591]] * 2)),
    ],
)
def test_level_set_kriging(kriging_distribution, cls, input_value, expected):
    """Check the criteria deriving from BaseLevelSet with GP regressor."""
    criterion = cls(kriging_distribution, 0.5)
    assert_almost_equal(criterion.func(input_value), expected)
    assert criterion._mc_size == 10000
    assert criterion._batch_size == 1


@pytest.mark.parametrize(
    ("cls", "input_value", "expected"),
    [
        (U, array([0.25]), array([0.35])),
        (EF, array([0.25]), array([0.41])),
        (EI, array([0.25]), array([1])),
        (U, array([[0.25]] * 2), array([[0.35]] * 2)),
        (EF, array([[0.25]] * 2), array([[0.41]] * 2)),
        (EI, array([[0.25]] * 2), array([[1]] * 2)),
    ],
)
def test_level_set_regressor(algo_distribution, cls, input_value, expected):
    """Check the criteria deriving from BaseLevelSet with non GP regressor."""
    criterion = cls(algo_distribution, 0.5)
    assert_almost_equal(criterion.func(input_value), expected, decimal=2)
    assert criterion._mc_size == 10000
    assert criterion._batch_size == 1


@pytest.mark.parametrize(
    ("cls", "expected", "options"),
    [
        (U, array([8.1571496e-08]), {}),
        (
            EF,
            0.9428005,
            {},
        ),
        (EI, array([0.8888888]), {}),
        (
            U,
            array([8.1571496e-08]),
            {"mc_size": 20000},
        ),
        (
            EF,
            0.942708,
            {"mc_size": 20000},
        ),
        (
            EI,
            array([0.8888888]),
            {"mc_size": 20000},
        ),
    ],
)
def test_level_set_parallel(kriging_distribution, cls, expected, options):
    """Check the parallelized criteria deriving from BaseLevelSet."""
    criterion = cls(
        regressor_distribution=kriging_distribution,
        output_value=0.5,
        batch_size=2,
        **options,
    )
    expected_mc_size = options.get("mc_size", 10000)
    input_value = array([[0.25], [0.25]])
    assert_almost_equal(criterion.func(input_value), expected)
    assert criterion._mc_size == expected_mc_size
    assert criterion._batch_size == 2


@pytest.mark.parametrize("cls", [EF, EI])
@pytest.mark.parametrize("distribution", ["algo_distribution", "kriging_distribution"])
def test_improvement_at_training_point(cls, distribution, request):
    """Check that the improvement criteria take predefined values a training point."""
    distribution = request.getfixturevalue(distribution)
    criterion = cls(distribution, 1)
    criterion._compute_standard_deviation = lambda x: array([0])
    assert_almost_equal(criterion.evaluate(array([0.0])), array([0.0]))


@pytest.mark.parametrize(
    ("distribution"),
    [
        ("kriging_distribution"),
        ("algo_distribution"),
    ],
)
def test_u_at_training_point(distribution, request):
    """Check that the U criterion at a training point is infinity."""
    distribution = request.getfixturevalue(distribution)
    u = U(distribution, 1)
    u._compute_standard_deviation = lambda x: array([0])
    infinity = nan_to_num(inf)

    # Case where mean=mu(x) and std=0 => U = abs(1-mu(x))/0
    assert_almost_equal(u.evaluate(array([0.0])), array([infinity]))

    # Case where mean=1 and std=0 => U = abs(1-1)/0 = 0/0
    u._compute_mean = lambda x: array([1])
    assert_almost_equal(u.evaluate(array([0.0])), array([infinity]))


@pytest.mark.parametrize("cls", [EF, EI])
def test_improvement_parallel_at_training_point(cls, kriging_distribution):
    """Check that the improvement criteria take predefined values a training point."""
    criterion = cls(kriging_distribution, 1, batch_size=2)
    criterion._compute_samples = lambda x: TypeError
    assert_almost_equal(criterion.evaluate(array([0.0, 0.0])), array([0.0]))


@pytest.mark.parametrize("std", [array([0]), TypeError])
def test_u_parallel_at_training_point(kriging_distribution, std):
    """Check that the discrepancy criterion take predefined values a training point."""
    u = U(kriging_distribution, 1, batch_size=2)
    u._compute_variance = lambda x: std
    assert_almost_equal(u.evaluate(array([0.0, 0.0])), nan_to_num(array([inf])))


@pytest.mark.parametrize("cls", [U, EI, EF])
def test_bad_parallel_regressor(algo_distribution, cls):
    """Check that parallelized criteria with non GP regressor lead to failure."""
    criterion = cls(algo_distribution, batch_size=2, output_value=1)

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "Parallelization with batch_size > 1 is not yet implemented "
            "for regressors that are not based on a random process."
        ),
    ):
        criterion.func(array([0.123]))

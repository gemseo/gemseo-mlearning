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

from gemseo_mlearning.active_learning.acquisition_criteria.maximum.ei import EI
from gemseo_mlearning.active_learning.acquisition_criteria.maximum.output import Output
from gemseo_mlearning.active_learning.acquisition_criteria.maximum.ucb import UCB


@pytest.mark.parametrize(
    ("cls", "options", "input_value", "expected"),
    [
        (EI, {}, array([0.0]), array([0.0])),
        (UCB, {}, array([0.0]), array([1.0])),
        (UCB, {"kappa": 3.0}, array([0.0]), array([1.0])),
        (Output, {}, array([0.0]), array([1.0])),
        (EI, {}, array([[0.0]] * 2), array([[0.0]] * 2)),
        (UCB, {}, array([[0.0]] * 2), array([[1.0]] * 2)),
        (UCB, {"kappa": 3.0}, array([[0.0]] * 2), array([[1.0]] * 2)),
        (Output, {}, array([[0.0]] * 2), array([[1.0]] * 2)),
    ],
)
def test_maximum_kriging_regressor(
    kriging_distribution, cls, options, input_value, expected
):
    """Check the criteria deriving from BaseMaximum with a Kriging distribution."""
    criterion = cls(kriging_distribution, **options)
    assert_almost_equal(criterion.func(input_value), expected)
    assert criterion._mc_size == 10000
    assert criterion._batch_size == 1


@pytest.mark.parametrize(
    ("cls", "options", "input_value", "expected"),
    [
        (EI, {}, array([0.123]), array([0.0])),
        (UCB, {}, array([0.123]), array([1.3609677])),
        (UCB, {"kappa": 3.0}, array([0.123]), array([1.8569516])),
        (Output, {}, array([0.123]), array([0.369])),
        (EI, {}, array([[0.123]] * 2), array([[0.0]] * 2)),
        (UCB, {}, array([[0.123]] * 2), array([[1.3609677]] * 2)),
        (UCB, {"kappa": 3.0}, array([[0.123]] * 2), array([[1.8569516]] * 2)),
        (Output, {}, array([[0.123]] * 2), array([[0.369]] * 2)),
    ],
)
def test_maximum_regressor(algo_distribution, options, cls, input_value, expected):
    """Check the criteria deriving from BaseMaximum with a RegressorDistribution."""
    criterion = cls(algo_distribution, **options)
    assert_almost_equal(criterion.func(input_value), expected)
    assert criterion._mc_size == 10000
    assert criterion._batch_size == 1


@pytest.mark.parametrize(
    ("options", "input_value", "expected"),
    [
        ({}, array([[1.0]] * 2), 0),
        ({}, array([[3]] * 2), 0.0668520),
        ({"mc_size": 50000}, array([[3]] * 2), 0.06647122),
    ],
)
def test_maximum_parallel_kriging_regressor(
    kriging_distribution, options, input_value, expected
):
    """Check the parallelized criteria deriving from BaseMaximum."""
    criterion = EI(kriging_distribution, batch_size=2, **options)
    assert_almost_equal(criterion.func(input_value), expected)
    expected_mc_size = options.get("mc_size", 10000)
    assert criterion._mc_size == expected_mc_size
    assert criterion._batch_size == 2


def test_improvement_parallel_at_training_point(kriging_distribution):
    """Check that the improvement criteria take predefined values a training point."""
    criterion = EI(kriging_distribution, 2)
    criterion._compute_samples = lambda x: TypeError
    assert_almost_equal(criterion.evaluate(array([0.0, 0.0])), 0)


@pytest.mark.parametrize(
    ("cls", "input_value"),
    [
        (EI, array([0.123])),
        (UCB, array([0.123])),
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

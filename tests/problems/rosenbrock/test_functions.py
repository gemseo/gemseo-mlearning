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
from __future__ import annotations

import pytest
from numpy import array
from numpy.testing import assert_almost_equal

from gemseo_mlearning.problems.rosenbrock.functions import compute_gradient
from gemseo_mlearning.problems.rosenbrock.functions import compute_output


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        (array([0.0, 1.0]), 101.0),
        (array([1.0, 1.0]), 0.0),
    ],
)
def test_compute_output(input_value, expected) -> None:
    """Check the output of the Rosenbrock function."""
    assert compute_output(input_value) == expected


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        (array([0.0, 1.0]), array([-2.0, 200.0])),
        (array([1.0, 1.0]), array([0.0, 0.0])),
    ],
)
def test_compute_gradient(input_value, expected) -> None:
    """Check the gradient of the Rosenbrock function."""
    assert_almost_equal(compute_gradient(input_value), expected)

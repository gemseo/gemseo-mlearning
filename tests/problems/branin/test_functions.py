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

from gemseo_mlearning.problems.branin.functions import compute_gradient
from gemseo_mlearning.problems.branin.functions import compute_output


def test_compute_output() -> None:
    """Check the output of the Branin function."""
    assert compute_output(array([0.0, 1.0])) == pytest.approx(17.5, abs=0.1)


def test_compute_gradient() -> None:
    """Check the gradient of the Branin function."""
    assert_almost_equal(
        compute_gradient(array([0.0, 1.0])), array([-327.3, -65.6]), decimal=1
    )

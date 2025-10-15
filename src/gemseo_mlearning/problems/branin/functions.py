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
"""The Branin function and its gradient."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from numpy import array
from numpy import cos
from numpy import pi
from numpy import sin

if TYPE_CHECKING:
    from numpy import ndarray

__A: Final[float] = 5.1 / (4 * pi**2)
__B: Final[float] = 5 / pi
__C: Final[float] = 10 - 10 / (8 * pi)


def compute_output(x: ndarray) -> float:
    """Compute the output of the Branin function.

    Args:
        x: The input values.

    Returns:
        The output value.
    """
    x0 = 15 * x[0] - 5
    return (15 * x[1] - __A * x0**2 + __B * x0 - 6) ** 2 + __C * cos(x0) + 10


def compute_gradient(x: ndarray) -> ndarray:
    """Compute the gradient of the Branin function.

    Args:
        x: The input values.

    Returns:
        The value of the gradient of the Branin function.
    """
    x0 = 15 * x[0] - 5
    tmp = 15 * x[1] - __A * x0**2 + __B * x0 - 6
    return array([-15 * (__C * sin(x0) + 2 * tmp * (2 * __A * x0 - __B)), 30 * tmp])

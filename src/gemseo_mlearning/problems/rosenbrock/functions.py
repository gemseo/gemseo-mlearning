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
"""The Rosenbrock function and its gradient."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from numpy import array

if TYPE_CHECKING:
    from numpy import ndarray

__A: Final[float] = 100.0


def compute_output(x: ndarray) -> float:
    """Compute the output of the Rosenbrock function.

    Args:
        x: The input value.

    Returns:
        The output value.
    """
    return (1 - x[0]) ** 2 + __A * (x[1] - x[0] ** 2) ** 2


def compute_gradient(x: ndarray) -> ndarray:
    """Compute the gradient of the Rosenbrock function.

    Args:
        x: The input value.

    Returns:
        The value of the gradient of the Rosenbrock function.
    """
    return array([
        4 * __A * (x[0] ** 3 - x[0] * x[1]) + 2 * (x[0] - 1),
        2 * __A * (x[1] - x[0] ** 2),
    ])

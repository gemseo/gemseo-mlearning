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
"""Utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.space import DesignSpace

if TYPE_CHECKING:
    from gemseo.space import RandomSpace


def create_design_space(random_space: RandomSpace) -> DesignSpace:
    """Create a design space from a random space.

    The bounds of a design variable are the limits of the support
    of the probability distribution of the corresponding random variable
    and its current value is the mean of this distribution.

    Args:
        random_space: The random space.

    Returns:
        The design space.
    """
    design_space = DesignSpace()
    reference_value = random_space.reference_value
    for name, variable in random_space.variables.items():
        design_space.add_real_variable(
            name,
            size=variable.size,
            lower_bound=variable.lower_bound,
            upper_bound=variable.upper_bound,
            value=reference_value[name],
        )

    return design_space

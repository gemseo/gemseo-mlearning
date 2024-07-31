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
"""The uncertain space used in the Branin use case."""

from __future__ import annotations

from openturns import Uniform

from gemseo_mlearning.problems.branin.branin_space import BraninSpace


def test_branin_space() -> None:
    """Check the Branin space."""
    space = BraninSpace()
    assert space.dimension == 2
    assert space.variable_names == ["x1", "x2"]
    for distribution in space.distributions.values():
        assert len(distribution.marginals) == 1
        distribution = distribution.marginals[0].distribution
        assert distribution.getParameter() == (0.0, 1.0)
        assert isinstance(distribution, Uniform)

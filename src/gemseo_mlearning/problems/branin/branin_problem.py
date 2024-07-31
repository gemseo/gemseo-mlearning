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
"""A problem connecting the Branin function with its uncertain space."""

from __future__ import annotations

from gemseo.algos.optimization_problem import OptimizationProblem

from gemseo_mlearning.problems.branin.branin_function import BraninFunction
from gemseo_mlearning.problems.branin.branin_space import BraninSpace


class BraninProblem(OptimizationProblem):
    """A problem connecting the Branin function with its uncertain space."""

    def __init__(self) -> None:  # noqa:D107
        super().__init__(BraninSpace())
        self.objective = BraninFunction()

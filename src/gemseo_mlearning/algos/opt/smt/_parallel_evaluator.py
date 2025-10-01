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
#    AUTHORS:
#       - Francois Gallard
"""Parallel evaluation on multiple processes or threads."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from numpy import array
from smt.applications.ego import Evaluator

if TYPE_CHECKING:
    from collections.abc import Callable
    from importlib.metadata import version

    from gemseo.typing import RealArray
    from packaging.version import parse

    if parse("2.8") > parse(version("smt")):
        from smt.utils.design_space import DesignSpace
    else:
        from smt.design_space import DesignSpace


class ParallelEvaluator(Evaluator):
    """Parallel evaluation on multiple processes or threads."""

    __n_processes: int
    """The number of processes or threads to use."""

    def __init__(self, n_processes: int) -> None:
        """
        Args:
            n_processes: The number of processes or threads to use.
        """  # noqa: D205 D212
        super().__init__()
        self.__n_processes = n_processes

    def run(
        self,
        function: Callable[[RealArray], float],
        input_values: RealArray,
        design_space: DesignSpace | None = None,
    ) -> RealArray:
        """Evaluate a function on a sample of input points.

        Args:
            function: The function to evaluate.
            input_values: The sample of input points.
            design_space: The design space.
                If ``None``, do not use it.

        Returns:
            The function evaluations on the sample of points.
        """
        if (n_processes := self.__n_processes) > 1:
            executor = CallableParallelExecution((function,), n_processes=n_processes)
            output_values = executor.execute(input_values)
        else:
            output_values = [function(input_value) for input_value in input_values]

        return array([output_values], dtype="float64").T

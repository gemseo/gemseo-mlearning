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
"""The Branin function as a discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.discipline import Discipline
from numpy import array
from numpy import concatenate

from gemseo_mlearning.problems.branin.functions import compute_gradient
from gemseo_mlearning.problems.branin.functions import compute_output

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.util.typing import StrKeyMapping


class BraninDiscipline(Discipline):
    """The Branin function as a discipline."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.input_grammar.update_from_names(["x1", "x2"])
        self.output_grammar.update_from_names(["y"])
        self.default_input_data.update({
            name: array([0.0]) for name in self.input_grammar.names
        })

    def _run(self, input_data: StrKeyMapping) -> None:
        # TODO(bump-gemseo): IO.data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or update_output_data(data)), write the inputs to input_data, and read input_data, output_data, get(name) or get_merged_data()  # noqa: E501
        inputs_array = concatenate([
            self.io.data[name] for name in self.io.input_grammar
        ])
        self.io.update_output_data({"y": array([compute_output(inputs_array)])})

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        # TODO(bump-gemseo): IO.data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or update_output_data(data)), write the inputs to input_data, and read input_data, output_data, get(name) or get_merged_data()  # noqa: E501
        inputs_array = concatenate([
            self.io.data[name] for name in self.io.input_grammar
        ])
        self.jac = {
            "y": {
                input_name: array([[derivative]])
                for input_name, derivative in zip(
                    self.io.input_grammar.names,
                    compute_gradient(inputs_array),
                    strict=False,
                )
            }
        }

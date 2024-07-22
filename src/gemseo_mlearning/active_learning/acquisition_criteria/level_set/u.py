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
"""U-function."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import inf
from numpy import nan_to_num

from gemseo_mlearning.active_learning.acquisition_criteria.level_set.base_level_set import (  # noqa: E501
    BaseLevelSet,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class U(BaseLevelSet):
    r"""The U-function.

    This acquisition criterion is expressed as:

    $$U[x] = \frac{|y-\mathbb{E}[Y(x)]|}{\mathbb{S}[Y(x)])}$$

    where $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$
    and $y$ is the model output value characterizing the level set.
    """

    MAXIMIZE: ClassVar[bool] = False

    def _compute_output(self, input_value: NumberArray) -> NumberArray:  # noqa: D102
        return nan_to_num(
            abs(self._output_value - self._compute_mean(input_value))
            / self._compute_standard_deviation(input_value),
            nan=nan_to_num(inf),
        )

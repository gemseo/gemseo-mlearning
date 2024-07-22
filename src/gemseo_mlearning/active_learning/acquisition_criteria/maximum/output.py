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
"""Output-based criterion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_mlearning.active_learning.acquisition_criteria.maximum._output import (
    Output as _Output,
)
from gemseo_mlearning.active_learning.acquisition_criteria.maximum.base_maximum import (
    BaseMaximum,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class Output(_Output, BaseMaximum):
    r"""Output-based criterion.

    This acquisition criterion is expressed as:

    $$E[x] = \mathbb{E}[Y(x)]$$

    where $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$.
    """

    def evaluate(self, input_data: NumberArray) -> NumberArray:  # noqa: D102
        return (
            self._regressor_distribution.compute_mean(input_data) / self._scaling_factor
        )

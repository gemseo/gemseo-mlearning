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
#
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Output standard deviation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_mlearning.active_learning.acquisition_criteria.exploration.base_exploration import (  # noqa: E501
    BaseExploration,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class StandardDeviation(BaseExploration):
    r"""Output standard deviation.

    This acquisition criterion is expressed as:

    $$\sigma[x] = \sqrt{\mathbb{E}[(Y(x)-\mathbb{E}[Y(x)])^2]}$$

    where $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$.
    """

    def _compute(self, input_value: NumberArray) -> NumberArray | float:  # noqa: D102
        return self._compute_standard_deviation(input_value) / self._scaling_factor

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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""Criterion to be maximized for estimating a maximum, without exploration.

Statistic:

$$E[x] = \mathbb{E}[Y(x)]$$

Bootstrap estimator:

$$\widehat{E}[x] = \frac{1}{B}\sum_{b=1}^B Y_b(x)$$
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion import (  # noqa: E501
    BaseAcquisitionCriterion,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class Expectation(BaseAcquisitionCriterion):
    """Criterion to be maximized for estimating a maximum, without exploration.

    This criterion is scaled by the output range.
    """

    def _compute_output(self, input_data: NumberArray) -> NumberArray:
        return self.algo_distribution.compute_mean(input_data) / self._scaling_factor

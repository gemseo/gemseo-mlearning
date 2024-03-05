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
r"""Criterion to be maximized for estimating a maximum, with exploration.

Statistic:

$$EI[x] = \mathbb{E}[\max(Y(x)-y_{\text{max}},0)]$$

where $y_{\text{max}}=\max_{1\leq i \leq n}~y^{(i)}$.

Bootstrap estimator:

$$\widehat{EI}[x] = \frac{1}{B}\sum_{b=1}^B \max(Y_b(x)-f_{\text{max}},0)$$
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion import (  # noqa: E501
    BaseAcquisitionCriterion,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class MaxExpectedImprovement(BaseAcquisitionCriterion):
    """Criterion to be maximized for estimating a maximum, with exploration.

    This criterion is scaled by the output range.
    """

    def _compute_output(self, input_data: NumberArray) -> NumberArray:
        data = self.algo_distribution.learning_set
        maximum_output = max(data.get_view(group_names=data.OUTPUT_GROUP).to_numpy())
        expected_improvement = self.algo_distribution.compute_expected_improvement(
            input_data, maximum_output, True
        )
        return expected_improvement / self._scaling_factor

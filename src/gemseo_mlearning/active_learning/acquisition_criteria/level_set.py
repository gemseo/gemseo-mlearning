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
r"""Criterion to be minimized for estimating a level set, with exploration.

Statistics:

$$EI[x] = \mathbb{E}[|q-Y(x)|]$$

where $q$ is a value provided by the user.

Bootstrap estimator:

$$\widehat{EI}[x] = \frac{1}{B}\sum_{b=1}^B |q-Y_b(x)|$$
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion import (  # noqa: E501
    BaseAcquisitionCriterion,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray

    from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (  # noqa: E501
        BaseRegressorDistribution,
    )


class LevelSet(BaseAcquisitionCriterion):
    """Criterion to be minimized for estimating a level set, with exploration."""

    value: float
    """The value of interest."""

    MAXIMIZE: ClassVar[bool] = False

    def __init__(
        self, algo_distribution: BaseRegressorDistribution, value: float
    ) -> None:
        """
        Args:
            value: A value of interest.
        """  # noqa: D205 D212 D415
        self.value = value
        super().__init__(algo_distribution)

    def _compute_output(self, input_data: NumberArray) -> NumberArray:
        mean = self.algo_distribution.compute_mean(input_data)
        std = self.algo_distribution.compute_standard_deviation(input_data)
        return abs(self.value - mean) / std

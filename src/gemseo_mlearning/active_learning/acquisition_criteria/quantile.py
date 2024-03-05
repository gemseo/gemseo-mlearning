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
r"""Quantile of the regression model.

Statistic:

$$EI[x] = \mathbb{E}[|q(\alpha)-Y(x)|]$$

where $q$ is a quantile with level $\alpha$.

Bootstrap estimator:

$$\widehat{EI}[x] = \frac{1}{B}\sum_{b=1}^B |q-Y_b(x)|$$
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import quantile

from gemseo_mlearning.active_learning.acquisition_criteria.level_set import LevelSet

if TYPE_CHECKING:
    from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (  # noqa: E501
        BaseRegressorDistribution,
    )


class Quantile(LevelSet):
    """Expected Improvement of the regression model for a given quantile."""

    def __init__(
        self, algo_distribution: BaseRegressorDistribution, level: float
    ) -> None:
        """
        Args:
            level: A quantile level.
        """  # noqa: D205 D212 D415
        dataset = algo_distribution.learning_set
        super().__init__(
            algo_distribution,
            quantile(dataset.get_view(group_names=dataset.OUTPUT_GROUP), level),
        )

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

Statistics:

.. math::

   EI[x] = E[|q(\alpha)-Y(x)|]

where :math:`q` is a quantile with level :math:`\alpha`.

Bootstrap estimator:

.. math::

   \widehat{EI}[x] = \frac{1}{B}\sum_{b=1}^B |q-Y_b(x)|
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import quantile

from gemseo_mlearning.adaptive.criteria.value.criterion import LimitState

if TYPE_CHECKING:
    from gemseo_mlearning.adaptive.distribution import MLRegressorDistribution


class Quantile(LimitState):
    """Expected Improvement of the regression model for a given quantile."""

    def __init__(
        self, algo_distribution: MLRegressorDistribution, level: float
    ) -> None:
        """
        Args:
            level: A quantile level.
        """  # noqa: D205 D212 D415
        dataset = algo_distribution.learning_set
        limit_state = quantile(
            dataset.get_view(group_names=dataset.OUTPUT_GROUP), level
        )
        super().__init__(algo_distribution, limit_state)

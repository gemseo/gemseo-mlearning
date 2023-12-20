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
r"""Combination of the expectation and standard deviation of the regression model.

Statistics:

.. math::

   E_sigma[x] = E[x] + \kappa \times sigma[x]

Estimator:

.. math::

   \widehat{E_sigma}[x] = \widehat{E}[x] + \kappa \times \widehat{sigma}[x]
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from gemseo_mlearning.adaptive.criterion import MLDataAcquisitionCriterion

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gemseo_mlearning.adaptive.distribution import MLRegressorDistribution


class MeanSigma(MLDataAcquisitionCriterion):
    """Combination of the expectation and standard deviation of the regression model.

    This criterion is scaled by the output range.
    """

    kappa: float
    """The factor associated with the standard deviation to increase the mean value."""

    def __init__(
        self, algo_distribution: MLRegressorDistribution, kappa: float
    ) -> None:
        """
        Args:
            kappa: A factor associated with the standard deviation
                to increase or decrease the mean value.
        """  # noqa: D205 D212 D415
        self.kappa = kappa
        super().__init__(algo_distribution)

    def _get_func(self) -> Callable[[NDArray[float]], float]:
        def func(input_data: NDArray[float]) -> float:
            """Evaluation function.

            Args:
                input_data: The model input data.

            Returns:
                The acquisition criterion value.
            """
            mean = self.algo_distribution.compute_mean(input_data)
            sigma = self.algo_distribution.compute_standard_deviation(input_data)
            return (mean + self.kappa * sigma) / self._scaling_factor

        return func

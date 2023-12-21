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
r"""Standard deviation of the regression model.

Statistics:

.. math::

   \sigma[x] = \sqrt{E[(Y(x)-E[Y(x)])^2]}

Bootstrap estimator:

.. math::

   \hat{\sigma}[x] = \sqrt{\frac{1}{B-1}\sum_{b=1}^B (Y_b(x)-\widehat{E}[x])^2}

where :math:`\widehat{E}[x]= \frac{1}{B}\sum_{b=1}^B Y_b(x)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from gemseo_mlearning.adaptive.criterion import MLDataAcquisitionCriterion

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StandardDeviation(MLDataAcquisitionCriterion):
    """Standard Deviation of the regression model.

    This criterion is scaled by the output range.
    """

    def _get_func(self) -> Callable[[NDArray[float]], float]:
        def func(input_data: NDArray[float]) -> float:
            """Evaluation function.

            Args:
                input_data: The model input data.

            Returns:
                The acquisition criterion value.
            """
            std = self.algo_distribution.compute_standard_deviation(input_data)
            return std / self._scaling_factor

        return func

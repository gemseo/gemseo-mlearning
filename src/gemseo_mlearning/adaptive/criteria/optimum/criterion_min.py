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
r"""Expected improvement for the minimum.

Statistics:

.. math::

   EI[x] = E[\max(y_{min}-Y(x),0)]

where :math:`y_{min}=\min_{1\leq i \leq n}~y^{(i)}`.

Bootstrap estimator:

.. math::

   \widehat{EI}[x] = \frac{1}{B}\sum_{b=1}^B \max(f_{min}-Y_b(x),0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from gemseo_mlearning.adaptive.criterion import MLDataAcquisitionCriterion

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MinExpectedImprovement(MLDataAcquisitionCriterion):
    """Expected Improvement of the regression model for the minimum.

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
            dataset = self.algo_distribution.learning_set
            minimum_output = min(
                dataset.get_view(group_names=dataset.OUTPUT_GROUP).to_numpy()
            )
            expected_improvement = self.algo_distribution.compute_expected_improvement(
                input_data, minimum_output
            )
            return expected_improvement / self._scaling_factor

        return func

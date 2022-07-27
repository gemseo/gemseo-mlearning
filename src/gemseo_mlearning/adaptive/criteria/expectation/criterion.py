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
r"""Expectation of the regression model.

Statistics:

.. math::

   E[x] = E[Y(x)]

Bootstrap estimator:

.. math::

   \widehat{E}[x] = \frac{1}{B}\sum_{b=1}^B Y_b(x)

"""
from __future__ import annotations

from numpy import ndarray

from gemseo_mlearning.adaptive.criterion import MLDataAcquisitionCriterion


class Expectation(MLDataAcquisitionCriterion):
    """Expectation of the regression model.".

    This criterion is scaled by the output range.
    """

    def _get_func(self):
        def func(input_data: ndarray) -> float:
            """Evaluation function.

            Args:
                input_data: The model input data.

            Returns:
                The acquisition criterion value.
            """
            return self.algo_distribution.compute_mean(input_data) / self.output_range

        return func

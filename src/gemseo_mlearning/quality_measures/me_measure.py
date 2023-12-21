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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""The maximum error measure to measure the quality of a regression algorithm.

The maximum error (ME) is defined by

.. math::

    \operatorname{ME}(\hat{y})=\max_{1\leq i \leq n}\|\hat{y}_i-y_i\|,

where :math:`\hat{y}` are the predictions and :math:`y` are the data points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.mlearning.quality_measures.error_measure import MLErrorMeasure

if TYPE_CHECKING:
    from gemseo.mlearning.regression.regression import MLRegressionAlgo
    from numpy import ndarray


class MEMeasure(MLErrorMeasure):
    """The maximum error measure for machine learning."""

    def __init__(  # noqa: D107
        self,
        algo: MLRegressionAlgo,
        fit_transformers: bool = False,
    ) -> None:
        super().__init__(algo, fit_transformers=fit_transformers)

    def _compute_measure(
        self,
        outputs: ndarray,
        predictions: ndarray,
        multioutput: bool = True,
    ) -> float | ndarray:
        if multioutput:
            return abs(outputs - predictions).max(0)
        return abs(outputs - predictions).max()

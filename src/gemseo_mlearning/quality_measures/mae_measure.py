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
r"""The mean absolute error to measure the quality of a regression algorithm.

The mean absolute error (MAE) is defined by

.. math::

    \operatorname{MAE}(\hat{y})=\frac{1}{n}\sum_{i=1}^n\|\hat{y}_i-y_i\|,

where :math:`\hat{y}` are the predictions and :math:`y` are the data points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.mlearning.quality_measures.error_measure import MLErrorMeasure
from sklearn.metrics import mean_absolute_error

if TYPE_CHECKING:
    from gemseo.mlearning.regression.regression import MLRegressionAlgo
    from numpy import ndarray


class MAEMeasure(MLErrorMeasure):
    """The mean absolute error measure for machine learning."""

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
        multioutput = "raw_values" if multioutput else "uniform_average"
        return mean_absolute_error(outputs, predictions, multioutput=multioutput)

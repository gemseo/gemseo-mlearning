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
#        :author: Matthias
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The gradient boosting for regression.

The gradient boosting model relies on the ``GradientBoostingRegressor`` class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.ensemble.GradientBoostingRegressor.html>`_.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from gemseo.mlearning.regression.regression import MLRegressionAlgo
from numpy import array
from numpy import ndarray
from sklearn.ensemble import GradientBoostingRegressor as SKLGradientBoosting

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.datasets.dataset import Dataset
    from gemseo.mlearning.core.ml_algo import TransformerType

LOGGER = logging.getLogger(__name__)


class GradientBoostingRegressor(MLRegressionAlgo):
    """Gradient boosting regression."""

    LIBRARY: Final[str] = "scikit-learn"
    SHORT_ALGO_NAME: ClassVar[str] = "GradientBoostingRegressor"

    def __init__(
        self,
        data: Dataset,
        transformer: Mapping[str, TransformerType] | None = None,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        n_estimators: int = 100,
        **parameters: Any,
    ) -> None:
        """
        Args:
            n_estimators: The number of boosting stages to perform.
        """  # noqa: D205 D212 D415
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            n_estimators=n_estimators,
            **parameters,
        )
        self.__algo = {"n_estimators": n_estimators, "parameters": parameters}
        self.algo = []

    def _fit(
        self,
        input_data: ndarray,
        output_data: ndarray,
    ) -> None:
        for _output_data in output_data.T:
            self.algo.append(
                SKLGradientBoosting(
                    n_estimators=self.__algo["n_estimators"],
                    **self.__algo["parameters"],
                )
            )
            self.algo[-1].fit(input_data, _output_data)

    def _predict(
        self,
        input_data: ndarray,
    ) -> ndarray:
        return array([algo.predict(input_data) for algo in self.algo]).T

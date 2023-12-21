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
"""Support vector machine for regression.

The support vector machine model relies on the :class:`SVR` class of ``sklearn``.
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
from sklearn.svm import SVR

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.datasets.dataset import Dataset
    from gemseo.mlearning.core.ml_algo import TransformerType

LOGGER = logging.getLogger(__name__)


class SVMRegressor(MLRegressionAlgo):
    """Support vector machine regressor."""

    LIBRARY: Final[str] = "scikit-learn"
    SHORT_ALGO_NAME: ClassVar[str] = "SVMRegression"

    def __init__(
        self,
        data: Dataset,
        transformer: Mapping[str, TransformerType] | None = None,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        kernel: str = "rbf",
        **parameters: Any,
    ) -> None:
        """
        Args:
            kernel: The kernel type to be used.
        """  # noqa: D205 D212 D415
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            kernel=kernel,
            **parameters,
        )
        self.__algo = {"kernel": kernel, "parameters": parameters}
        self.algo = []

    def _fit(
        self,
        input_data: ndarray,
        output_data: ndarray,
    ) -> None:
        for _output_data in output_data.T:
            self.algo.append(
                SVR(
                    kernel=self.__algo["kernel"],
                    **self.__algo["parameters"],
                )
            )
            self.algo[-1].fit(input_data, _output_data)

    def _predict(
        self,
        input_data: ndarray,
    ) -> ndarray:
        return array([algo.predict(input_data) for algo in self.algo]).T

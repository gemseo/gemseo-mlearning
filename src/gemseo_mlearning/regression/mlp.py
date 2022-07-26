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
"""Interface to the OpenTURNS' multilayer perceptron (MLP)."""
from __future__ import annotations

import logging
from typing import ClassVar
from typing import Iterable
from typing import Mapping

import sklearn.neural_network
from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.utils.python_compatibility import Final
from numpy import ndarray

LOGGER = logging.getLogger(__name__)


class MLPRegressor(MLRegressionAlgo):
    """MultiLayer perceptron (MLP)."""

    LIBRARY: Final[str] = "scikit-learn"
    SHORT_ALGO_NAME: ClassVar[str] = "MLP"

    def __init__(
        self,
        data: Dataset,
        transformer: Mapping[str, TransformerType] | None = None,
        input_names: Iterable[str] = None,
        output_names: Iterable[str] = None,
        hidden_layer_sizes: tuple[int] = (100,),
        **parameters,
    ) -> None:
        """# noqa: D205 D212 D415
        Args:
            hidden_layer_sizes: The number of neurons per hidden layer.
        """
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            hidden_layer_sizes=hidden_layer_sizes,
            **parameters,
        )
        self.algo = sklearn.neural_network.MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, **parameters
        )

    def _fit(
        self,
        input_data: ndarray,
        output_data: ndarray,
    ) -> None:
        self.algo.fit(input_data, output_data)

    def _predict(
        self,
        input_data: ndarray,
    ) -> ndarray:
        output_data = self.algo.predict(input_data)
        if output_data.ndim == 1:
            return output_data[:, None]

        return output_data

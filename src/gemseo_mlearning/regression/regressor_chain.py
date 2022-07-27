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
"""A model chaining regression models.

During the training stage,
the first regression model learns the learning dataset,
the second regression model learns the learning error of the first regression model,
and the $i$-th regression model learns the learning error of its predecessor.

During the prediction stage,
the different regression models are evaluated from a new input data
and the sum of their output data is returned.
"""
from __future__ import annotations

import logging
from collections import namedtuple
from typing import ClassVar
from typing import Iterable
from typing import Mapping

from gemseo.core.dataset import Dataset
from gemseo.mlearning.api import create_regression_model
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from numpy import ndarray

LOGGER = logging.getLogger(__name__)

_AlgoDefinition = namedtuple("AlgoDefinition", "name,transformer,parameters")


class RegressorChain(MLRegressionAlgo):
    """Chain regression."""

    SHORT_ALGO_NAME: ClassVar[str] = "RegressorChain"

    def __init__(  # noqa: D107
        self,
        data: Dataset,
        transformer: Mapping[str, TransformerType] | None = None,
        input_names: Iterable[str] = None,
        output_names: Iterable[str] = None,
        **parameters,
    ) -> None:
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            **parameters,
        )
        self.__algos = []

    def add_algo(
        self,
        name: str,
        transformer: Mapping[str, TransformerType] | None = None,
        **parameters,
    ) -> None:
        """Add a new regression algorithm in the chain.

        Args:
            name: The name of the regression algorithm.
            transformer: The strategies to transform the variables.
                The values are instances of :class:`.Transformer`
                while the keys are the names of
                either the variables
                or the groups of variables,
                e.g. "inputs" or "outputs" in the case of the regression algorithms.
                If a group is specified,
                the :class:`.Transformer` will be applied
                to all the variables of this group.
                If ``None``, do not transform the variables.
            **parameters: The parameters of the regression algorithm
        """
        self.__algos.append(
            create_regression_model(
                name,
                self.learning_set,
                transformer=transformer,
                **parameters,
            )
        )

    def _fit(
        self,
        input_data: ndarray,
        output_data: ndarray,
    ) -> None:
        for index, algo in enumerate(self.__algos):
            algo._fit(input_data, output_data)
            output_data -= algo._predict(input_data)
            self.__algos[index] = algo

    def _predict(
        self,
        input_data: ndarray,
    ) -> ndarray:
        output_data = 0
        for algo in self.__algos:
            output_data += algo._predict(input_data)

        return output_data

    def _predict_jacobian(
        self,
        input_data: ndarray,
    ) -> ndarray:
        jacobian_data = 0
        for algo in self.__algos:
            jacobian_data += algo._predict_jacobian(input_data)

        return jacobian_data

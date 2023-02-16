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
"""Gaussian process regression model from OpenTURNS."""
from __future__ import annotations

from typing import ClassVar
from typing import Iterable

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import DataType
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.utils.data_conversion import (
    concatenate_dict_of_arrays_to_array,
)
from gemseo.utils.python_compatibility import Final
from numpy import array
from numpy import atleast_2d
from numpy import diag
from numpy import ndarray
from openturns import ConstantBasisFactory
from openturns import KrigingAlgorithm
from openturns import Point
from openturns import ResourceMap
from openturns import SquaredExponential
from openturns import TensorizedCovarianceModel


class OTGaussianProcessRegressor(MLRegressionAlgo):
    """Gaussian process regression model from OpenTURNS."""

    LIBRARY: Final[str] = "OpenTURNS"
    SHORT_ALGO_NAME: ClassVar[str] = "GPR"

    MAX_SIZE_FOR_LAPACK: ClassVar[int] = 100
    """The maximum size of the learning dataset to use LAPACK as linear algebra library.

    Use HMAT otherwise.
    """

    HMATRIX_ASSEMBLY_EPSILON: ClassVar[float] = 1e-5
    """The epsilon used for the assembly of the H-matrix.

    Used when ``use_hmat`` is ``True``.
    """

    HMATRIX_RECOMPRESSION_EPSILON: ClassVar[float] = 1e-4
    """The epsilon used for the recompression of the H-matrix.

    Used when ``use_hmat`` is ``True``.
    """

    def __init__(
        self,
        data: Dataset,
        transformer: TransformerType | None = None,
        input_names: Iterable[str] = None,
        output_names: Iterable[str] = None,
        use_hmat: bool = None,
    ) -> None:
        """# noqa: D205 D212 D415
        Args:
            use_hmat: Whether to use the HMAT or LAPACK as linear algebra method.
                If ``None``,
                use HMAT when the learning size is greater
                than :attr:`MAX_SIZE_FOR_LAPACK`.
        """
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            use_hmat=use_hmat,
        )
        self.__use_hmat = None
        if use_hmat is None:
            self.use_hmat = len(data) > self.MAX_SIZE_FOR_LAPACK
        else:
            self.use_hmat = use_hmat

    @property
    def use_hmat(self) -> bool:
        """Whether to use the HMAT linear algebra method or LAPACK."""
        return self.__use_hmat

    @use_hmat.setter
    def use_hmat(self, use_hmat: bool) -> None:
        self.__use_hmat = use_hmat
        if use_hmat:
            linear_algebra_method = "HMAT"
            ResourceMap.SetAsScalar(
                "HMatrix-AssemblyEpsilon", self.HMATRIX_ASSEMBLY_EPSILON
            )
            ResourceMap.SetAsScalar(
                "HMatrix-RecompressionEpsilon", self.HMATRIX_RECOMPRESSION_EPSILON
            )
        else:
            linear_algebra_method = "LAPACK"
        ResourceMap.SetAsString("KrigingAlgorithm-LinearAlgebra", linear_algebra_method)

    def _fit(self, input_data: ndarray, output_data: ndarray) -> None:
        input_dimension = input_data.shape[1]
        output_dimension = output_data.shape[1]
        covariance_models = [
            SquaredExponential([0.1] * input_dimension, [1.0])
            for _ in range(output_dimension)
        ]
        if output_dimension == 1:
            covariance_model = covariance_models[0]
        else:
            covariance_model = TensorizedCovarianceModel(covariance_models)

        algo = KrigingAlgorithm(
            input_data,
            output_data,
            covariance_model,
            ConstantBasisFactory(input_dimension).build(),
        )
        algo.run()
        self.algo = algo.getResult()

    def _predict(self, input_data: ndarray) -> ndarray:
        return atleast_2d(self.algo.getConditionalMean(input_data))

    def predict_std(self, input_data: DataType) -> ndarray:
        """Predict the standard deviation from input data.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            output_data: The output data with shape (n_samples, n_outputs).
        """
        if isinstance(input_data, dict):
            input_data = concatenate_dict_of_arrays_to_array(
                input_data, self.input_names
            )

        one_dim = input_data.ndim == 1
        input_data = atleast_2d(input_data)
        inputs = self.learning_set.INPUT_GROUP
        if inputs in self.transformer:
            input_data = self.transformer[inputs].transform(input_data)

        output_data = array(
            [
                diag(self.algo.getConditionalCovariance(input_datum)).tolist()
                for input_datum in input_data
            ]
        )

        if one_dim:
            return output_data[0]

        return output_data

    def _predict_jacobian(self, input_data: ndarray) -> ndarray:
        gradient = self.algo.getMetaModel().gradient
        return array([array(gradient(Point(data))).T for data in input_data])

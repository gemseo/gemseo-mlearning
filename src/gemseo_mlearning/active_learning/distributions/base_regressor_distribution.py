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
"""Base class for regressor distributions."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo.mlearning.data_formatters.regression_data_formatters import (
    RegressionDataFormatters,
)
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.core.algos.ml_algo import DataType
    from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
    from gemseo.typing import NumberArray


class BaseRegressorDistribution(metaclass=ABCGoogleDocstringInheritanceMeta):
    """The distribution of a regressor."""

    algo: BaseRegressor
    """The regression model."""

    _samples: list[int]
    """The indices of the learning samples in the learning dataset."""

    _transform_input_group: bool
    """Whether to transform the input group."""

    _transform_output_group: bool
    """Whether to transform the output group."""

    _input_variables_to_transform: list[str]
    """The names of the input variables to be transformed."""

    _output_variables_to_transform: list[str]
    """The names of the output variables to be transformed."""

    def __init__(self, regressor: BaseRegressor) -> None:
        """
        Args:
            regressor: A regression model.
        """  # noqa: D205 D212 D415
        self.algo = regressor
        self._samples = []
        self._transform_input_group = self.algo._transform_input_group
        self._transform_output_group = self.algo._transform_output_group
        self._input_variables_to_transform = self.algo._input_variables_to_transform
        self._output_variables_to_transform = self.algo._output_variables_to_transform

    @property
    def learning_set(self) -> IODataset:
        """The learning dataset used by the regressor."""
        return self.algo.learning_set

    @property
    def input_names(self) -> list[str]:
        """The input names of the regressor."""
        return self.algo.input_names

    @property
    def output_names(self) -> list[str]:
        """The output names of the regressor."""
        return self.algo.output_names

    @property
    def output_dimension(self) -> int:
        """The output dimension of the regressor."""
        return self.algo.output_dimension

    def learn(self, samples: list[int] | None = None) -> None:
        """Train the regressor from the learning dataset.

        Args:
            samples: The indices of the learning samples.
                If `None`, use the whole learning dataset
        """
        self._samples = samples or range(len(self.learning_set))
        self.algo.learn(self._samples)

    def predict(
        self,
        input_data: DataType,
    ) -> DataType:
        """Compute the output value(s) of the regressor.

        The user can specify the input data either as a NumPy array,
        e.g. `array([1., 2., 3.])`
        or as a dictionary,
        e.g. `{"a": array([1.]), "b": array([2., 3.])}`.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.

        Returns:
            The output value(s) of the regressor.
        """
        return self.algo.predict(input_data)

    @abstractmethod
    def compute_confidence_interval(
        self,
        input_data: DataType,
        level: float = 0.95,
    ) -> (
        tuple[
            dict[str, NumberArray],
            dict[str, NumberArray],
            tuple[NumberArray, NumberArray],
        ]
        | None
    ):
        """Compute the lower bounds and upper bounds of the regressor.

        The user can specify the input data either as a NumPy array,
        e.g. `array([1., 2., 3.])`
        or as a dictionary,
        e.g.  `{"a": array([1.]), "b": array([2., 3.])}`.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.
            level: A quantile level.

        Returns:
            The lower and upper bound values of the regressor.
        """

    @abstractmethod
    def compute_mean(
        self,
        input_data: DataType,
    ) -> DataType:
        """Compute the output mean of the regressor.

        The user can specify the input data either as a NumPy array,
        e.g. `array([1., 2., 3.])`
        or as a dictionary,
        e.g.  `{"a": array([1.]), "b": array([2., 3.])}`.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.

        Returns:
            The output mean of the regressor.
        """

    @abstractmethod
    def compute_variance(
        self,
        input_data: DataType,
    ) -> DataType:
        """Compute the output variance of the regressor.

        The user can specify the input data either as a NumPy array,
        e.g. `array([1., 2., 3.])`
        or as a dictionary,
        e.g.  `{"a": array([1.]), "b": array([2., 3.])}`.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.

        Returns:
            The output variance of the regressor.
        """

    @RegressionDataFormatters.format_dict
    @RegressionDataFormatters.format_samples()
    def compute_standard_deviation(
        self,
        input_data: DataType,
    ) -> DataType:
        """Compute the output standard deviation of the regressor.

        The user can specify the input data either as a NumPy array,
        e.g. `array([1., 2., 3.])`
        or as a dictionary,
        e.g.  `{"a": array([1.]), "b": array([2., 3.])}`.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.

        Returns:
            The output standard deviation  of the regressor.
        """
        return self.compute_variance(input_data) ** 0.5

    def change_learning_set(self, learning_set: Dataset) -> None:
        """Re-train the regressor from a new learning set.

        Args:
            learning_set: The new learning set.
        """
        self.algo.learning_set = learning_set
        self.learn()

    @abstractmethod
    def compute_samples(
        self,
        input_data: NumberArray,
        n_samples: int,
    ) -> NumberArray:
        """Generate samples from the random process.

        Args:
            input_data: The $N$ input points of dimension $d$
                at which to observe the random process;
                shaped as `(N, d)`.
            n_samples: The number of samples `M`.

        Returns:
            The output samples per output dimension shaped as `(N, M, p)`
            where `p` is the output dimension.
        """

    @abstractmethod
    def compute_covariance(
        self,
        input_data: NumberArray,
    ) -> NumberArray:
        r"""Compute the output covariance of the regressor.

        Args:
            input_data: The $N$ input points of dimension $d$
                at which to compute the covariance;
                shaped as $(N, d)$.

        Returns:
            The posterior covariance matrix at the input points
            of shape $(Np, Np)$
            with $p$ the output dimension.
            The covariance between
            the $k$-th output
            at the $i$-th input point
            and the $l$-th output
            at the $j$-th input point
            is located at
            the $m$-th line and $n$-th column
            with $m=ip+k$, $n=jp+l$,
            $i,j\in\{0,\ldots,N-1\}$
            and $k,l\in\{0,\ldots,p-1\}$.
        """

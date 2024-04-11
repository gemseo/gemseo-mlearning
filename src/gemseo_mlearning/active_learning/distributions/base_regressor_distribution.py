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
"""This module defines the notion of distribution of a machine learning algorithm.

Once a
[BaseMLAlgo][gemseo.mlearning.core.ml_algo.BaseMLAlgo]
has been trained,
assessing its quality is important before using it.

One can not only measure its global quality
(e.g. from a
[MLQualityMeasure][gemseo.mlearning.quality_measures.quality_measure.MLQualityMeasure]
)
but also its local one.

The
[BaseRegressorDistribution][gemseo_mlearning.active_learning.distributions.base_regressor_distribution.BaseRegressorDistribution]
class addresses the latter case,
by quantifying the robustness of a machine learning algorithm to a learning point.
The more robust it is,
the less variability it has around this point.

???+ note

    For now, one does not consider any [BaseMLAlgo][gemseo.mlearning.core.ml_algo.BaseMLAlgo]
    but instances of
    class which is built from a
    [MLSupervisedAlgo][gemseo.mlearning.core.supervised.MLSupervisedAlgo].


The
[BaseRegressorDistribution][gemseo_mlearning.active_learning.distributions.base_regressor_distribution.BaseRegressorDistribution]
can be particularly useful to:

- study the robustness of an [BaseMLAlgo][gemseo.mlearning.core.ml_algo.BaseMLAlgo]
  w.r.t. learning dataset elements,
- evaluate acquisition criteria for active learning purposes
  (see
  [BaseAcquisitionCriterion][gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion.BaseAcquisitionCriterion]
  ),
- etc.

The abstract
[BaseRegressorDistribution][gemseo_mlearning.active_learning.distributions.base_regressor_distribution.BaseRegressorDistribution]
class is derived into two classes:

- [KrigingDistribution][gemseo_mlearning.active_learning.distributions.kriging_distribution.KrigingDistribution]
  :
  the
  [BaseMLRegressionAlgo][gemseo.mlearning.regression.regression.BaseMLRegressionAlgo]
  is a Kriging model
  and this assessor takes advantage of the underlying Gaussian stochastic process,
- [RegressorDistribution][gemseo_mlearning.active_learning.distributions.regressor_distribution.RegressorDistribution]:
  this class is based on sampling methods,
  such as bootstrap,
  cross-validation
  or leave-one-out.
"""  # noqa: E501

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo.mlearning.regression import regression
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.core.ml_algo import DataType
    from gemseo.mlearning.regression.regression import BaseMLRegressionAlgo
    from gemseo.typing import NumberArray


class BaseRegressorDistribution(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Distribution related to a regression model."""

    algo: BaseMLRegressionAlgo
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

    def __init__(self, algo: BaseMLRegressionAlgo) -> None:
        """
        Args:
            algo: A regression model.
        """  # noqa: D205 D212 D415
        self.algo = algo
        self._samples = []
        self._transform_input_group = self.algo._transform_input_group
        self._transform_output_group = self.algo._transform_output_group
        self._input_variables_to_transform = self.algo._input_variables_to_transform
        self._output_variables_to_transform = self.algo._output_variables_to_transform

    @property
    def learning_set(self) -> IODataset:
        """The learning dataset used by the original machine learning algorithm."""
        return self.algo.learning_set

    @property
    def input_names(self) -> list[str]:
        """The names of the original machine learning algorithm inputs."""
        return self.algo.input_names

    @property
    def output_names(self) -> list[str]:
        """The names of the original machine learning algorithm outputs."""
        return self.algo.output_names

    @property
    def output_dimension(self) -> int:
        """The dimension of the machine learning output space."""
        return self.algo.output_dimension

    def learn(self, samples: list[int] | None = None) -> None:
        """Train the machine learning algorithm from the learning dataset.

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
        """Predict the output of the original machine learning algorithm.

        The user can specify the input data either as a NumPy array,
        e.g. `array([1., 2., 3.])`
        or as a dictionary,
        e.g. `{"a": array([1.]), "b": array([2., 3.])}`.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.

        Returns:
            The predicted output data.
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
        """Predict the lower bounds and upper bounds from input data.

        The user can specify the input data either as a NumPy array,
        e.g. `array([1., 2., 3.])`
        or as a dictionary,
        e.g.  `{"a": array([1.]), "b": array([2., 3.])}`.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.
            level: A quantile level.

        Returns:
            The lower and upper bound values.
        """

    @abstractmethod
    def compute_mean(
        self,
        input_data: DataType,
    ) -> DataType:
        """Compute the mean from input data.

        The user can specify the input data either as a NumPy array,
        e.g. `array([1., 2., 3.])`
        or as a dictionary,
        e.g.  `{"a": array([1.]), "b": array([2., 3.])}`.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.

        Returns:
            The mean value.
        """

    @abstractmethod
    def compute_variance(
        self,
        input_data: DataType,
    ) -> DataType:
        """Compute the variance from input data.

        The user can specify the input data either as a NumPy array,
        e.g. `array([1., 2., 3.])`
        or as a dictionary,
        e.g.  `{"a": array([1.]), "b": array([2., 3.])}`.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.

        Returns:
            The variance value.
        """

    @regression.BaseMLRegressionAlgo.DataFormatters.format_dict
    @regression.BaseMLRegressionAlgo.DataFormatters.format_samples
    def compute_standard_deviation(
        self,
        input_data: DataType,
    ) -> DataType:
        """Compute the standard deviation from input data.

        The user can specify the input data either as a NumPy array,
        e.g. `array([1., 2., 3.])`
        or as a dictionary,
        e.g.  `{"a": array([1.]), "b": array([2., 3.])}`.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.

        Returns:
            The standard deviation value.
        """
        return self.compute_variance(input_data) ** 0.5

    @abstractmethod
    def compute_expected_improvement(
        self, input_data: DataType, fopt: float, maximize: bool = False
    ) -> DataType:
        """Compute the expected improvement from input data.

        The user can specify the input data either as a NumPy array,
        e.g. `array([1., 2., 3.])`
        or as a dictionary,
        e.g.  `{"a": array([1.]), "b": array([2., 3.])}`.

        The output data type will be consistent with the input data type.

        Args:
            input_data: The input data.
            fopt: The current optimum value.
            maximize: The type of optimum to seek.

        Returns:
            The expected improvement value.
        """

    def change_learning_set(self, learning_set: Dataset) -> None:
        """Re-train the machine learning algorithm relying on the initial learning set.

        Args:
            learning_set: The new learning set.
        """
        self.algo.learning_set = learning_set
        self.learn()

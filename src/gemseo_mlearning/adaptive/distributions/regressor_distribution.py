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
#
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Universal distribution for regression models.

A :class:`.RegressorDistribution` samples a :class:`.MLSupervisedAlgo`,
by learning new versions of the latter from subsets of the original learning dataset.

These new :class:`.MLAlgo` instances are based on sampling methods,
such as bootstrap, cross-validation or leave-one-out.

Sampling a :class:`.MLAlgo` can be particularly useful to:

- study the robustness of a :class:`.MLAlgo` w.r.t. learning dataset elements,
- estimate infill criteria for adaptive learning purposes,
- etc.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar
from typing import Final

from gemseo.mlearning.regression import regression
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from numpy import array
from numpy import array_split
from numpy import atleast_2d
from numpy import delete as npdelete
from numpy import dot
from numpy import exp
from numpy import maximum
from numpy import ndarray
from numpy import ones
from numpy import quantile
from numpy import stack
from numpy import sum as npsum
from numpy import unique
from numpy.random import default_rng
from scipy.spatial.distance import euclidean

from gemseo_mlearning.adaptive.distribution import MLRegressorDistribution

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset
    from gemseo.mlearning.core.ml_algo import DataType
    from gemseo.mlearning.regression.regression import MLRegressionAlgo

LOGGER = logging.getLogger(__name__)


class RegressorDistribution(MLRegressorDistribution):
    """Distribution related to a regression algorithm."""

    method: str
    """The resampling method."""

    size: int
    """The size of the resampling set."""

    weights: list[Callable[[ndarray], float]]
    """The weight functions related to the sub-algorithms.

    A weight function computes a weight from an input data array.
    """

    N_BOOTSTRAP: ClassVar[int] = 100
    N_FOLDS: ClassVar[int] = 5
    CROSS_VALIDATION: Final[str] = "cv"
    BOOTSTRAP: Final[str] = "b"
    LOO: Final[str] = "loo"

    def __init__(
        self,
        algo: MLRegressionAlgo,
        bootstrap: bool = True,
        loo: bool = False,
        size: int | None = None,
    ) -> None:
        """
        Args:
            bootstrap: The resampling method.
                If True, use bootstrap resampling.
                Otherwise, use cross-validation resampling.
            loo: The leave-One-Out sub-method, when bootstrap is False.
                If False, use parameterized cross-validation,
                Otherwise use leave-one-out.
            size: The size of the resampling set,
                i.e. the number of times the machine learning algorithm is rebuilt.
                If ``None``, use the default size
                for bootstrap (:attr:`.MLAlgoSampler.N_BOOTSTRAP`)
                and cross-validation (:attr:`.MLAlgoSampler.N_FOLDS`).
        """  # noqa: D205 D212 D415
        if bootstrap:
            self.method = self.BOOTSTRAP
            self.size = size or self.N_BOOTSTRAP
        else:
            if loo:
                self.method = self.LOO
                self.size = len(algo.learning_set)
            else:
                self.method = self.CROSS_VALIDATION
                self.size = size or self.N_FOLDS
        self.algos = [
            algo.__class__(
                data=algo.learning_set, transformer=algo.transformer, **algo.parameters
            )
            for _ in range(self.size)
        ]
        self.weights = []
        super().__init__(algo)

    def learn(  # noqa: D102
        self,
        samples: list[int] | None = None,
    ) -> None:
        self.weights = []
        super().learn(samples)
        if self.method in [self.CROSS_VALIDATION, self.LOO]:
            n_folds = self.size
            folds = array_split(self._samples, n_folds)
        for index, algo in enumerate(self.algos):
            if self.method == self.BOOTSTRAP:
                new_samples = unique(
                    default_rng(1).choice(self._samples, len(self._samples))
                )
                other_samples = list(set(self._samples) - set(new_samples))
                self.weights.append(self.__weight_function(other_samples))
            else:
                fold = folds[index]
                new_samples = npdelete(self._samples, fold)
                other_samples = list(set(self._samples) - set(new_samples))
                self.weights.append(self.__weight_function(other_samples))

            algo.learn(new_samples)

    def __weight_function(self, indices: list[int]) -> Callable[[ndarray], float]:
        """Return a function evaluating the weights at an input vector.

        The weights are w.r.t. the input vectors of the samples
        skipped at iteration indices.

        Args:
            indices: The indices of the samples
                removed from the learning dataset during the training phase.

        Returns:
            The weight function returning a weight from a 1D input array.
        """
        dat = self.learning_set.get_view(
            group_names=self.learning_set.INPUT_GROUP
        ).to_numpy()
        all_indices = set(self._samples)
        rho = max(
            min(euclidean(dat[id1], dat[id2]) for id2 in all_indices - {id1})
            for id1 in all_indices
        )
        in_grp = self.learning_set.INPUT_GROUP

        def weight(
            input_data: ndarray,
        ) -> float:
            """Weight function returning a weight when a 1D input array is passed.

            Args:
                input_data: A 1D input array.

            Returns:
                A weight related to the input array.
            """
            if isinstance(input_data, dict):
                input_data = concatenate_dict_of_arrays_to_array(
                    input_data, self.input_names
                )
            only_one_element = input_data.ndim == 1
            input_data = atleast_2d(input_data)
            distance = ones(input_data.shape[0])
            for index in indices:
                index_data = self.learning_set.get_view(group_names=in_grp).to_numpy()[
                    index
                ]
                for index, value in enumerate(input_data):
                    term = 1 - exp(-(euclidean(index_data, value) ** 2) / rho**2)
                    distance[index] *= term
            if only_one_element:
                distance = distance[0]
            return distance

        return weight

    def predict_members(self, input_data: DataType) -> DataType:
        """Predict the output value with the different machine learning algorithms.

        After prediction, the method stacks the results.

        Args:
            input_data: The input data,
                specified as either as a numpy.array or as dictionary of numpy.array
                indexed by inputs names.
                The numpy.array can be either a (d,) array
                representing a sample in dimension d,
                or a (M, d) array representing M samples in dimension d.

        Returns:
            The output data (dimension p) of N machine learning algorithms.
                If input_data.shape == (d,), then output_data.shape == (N, p).
                If input_data.shape == (M,d), then output_data;shape == (N,M,p).
        """
        predictions = [algo.predict(input_data) for algo in self.algos]
        if isinstance(input_data, dict):
            return {
                name: stack([prediction[name] for prediction in predictions])
                for name in predictions[0]
            }
        return stack(predictions)

    def compute_confidence_interval(  # noqa: D102
        self,
        input_data: DataType,
        level: float = 0.95,
    ) -> tuple[dict[str, ndarray], dict[str, ndarray], tuple[ndarray, ndarray]] | None:
        level = (1.0 - level) / 2.0
        predictions = self.predict_members(input_data)
        if isinstance(predictions, dict):
            lower = {
                name: quantile(value, level, axis=0)
                for name, value in predictions.items()
            }
            upper = {
                name: quantile(value, 1 - level, axis=0)
                for name, value in predictions.items()
            }
        else:
            lower = quantile(predictions, level, axis=0)
            upper = quantile(predictions, 1 - level, axis=0)
        return lower, upper

    def _evaluate_weights(self, input_data: ndarray) -> ndarray:
        """Evaluate weights.

        Args:
            input_data: The input data with shape (size, n_input_data)

        Returns:
            The weights.
        """
        weights = array([func(input_data) for func in self.weights])
        return weights / npsum(weights, 0)

    @regression.MLRegressionAlgo.DataFormatters.format_dict
    @regression.MLRegressionAlgo.DataFormatters.format_samples
    def compute_mean(self, input_data: DataType) -> DataType:  # noqa: D102
        predictions = self.predict_members(input_data)
        weights = self._evaluate_weights(input_data)
        return self.__average(weights, predictions)

    @regression.MLRegressionAlgo.DataFormatters.format_dict
    @regression.MLRegressionAlgo.DataFormatters.format_samples
    def compute_variance(self, input_data: DataType) -> DataType:  # noqa: D102
        predictions = self.predict_members(input_data)
        weights = self._evaluate_weights(input_data)
        term1 = self.__average(weights, predictions**2)
        term2 = self.__average(weights, predictions) ** 2
        return term1 - term2

    @regression.MLRegressionAlgo.DataFormatters.format_dict
    @regression.MLRegressionAlgo.DataFormatters.format_samples
    def compute_expected_improvement(  # noqa: D102
        self,
        input_data: DataType,
        fopt: float,
        maximize: bool = False,
    ) -> DataType:
        predictions = self.predict_members(input_data)
        weights = self._evaluate_weights(input_data)
        if maximize:
            expected_improvement = maximum(predictions - fopt, 0.0)
        else:
            expected_improvement = maximum(fopt - predictions, 0.0)
        return self.__average(weights, expected_improvement)

    def __average(
        self,
        weights: ndarray,
        data: ndarray,
    ) -> ndarray:
        """Return averaged data.

        Args:
            weights: The weights.
            data: The data from which to get the average.

        Returns:
            The averaged value.
        """
        return array([
            dot(weights[:, index], data[:, index, :]) for index in range(data.shape[1])
        ])

    def change_learning_set(self, learning_set: Dataset) -> None:  # noqa: D102
        for algo in self.algos:
            algo.learning_set = learning_set
        super().change_learning_set(learning_set)

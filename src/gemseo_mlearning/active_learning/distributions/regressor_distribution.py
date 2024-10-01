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
"""Universal regressor distribution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar
from typing import Final

from gemseo.mlearning.data_formatters.regression_data_formatters import (
    RegressionDataFormatters,
)
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from numpy import array
from numpy import array_split
from numpy import atleast_2d
from numpy import delete as npdelete
from numpy import dot
from numpy import exp
from numpy import ones
from numpy import quantile
from numpy import stack
from numpy import sum as npsum
from numpy import unique
from numpy.random import default_rng
from scipy.spatial.distance import euclidean

from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (
    BaseRegressorDistribution,
)

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset
    from gemseo.mlearning.core.algos.ml_algo import DataType
    from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
    from gemseo.typing import NumberArray


class RegressorDistribution(BaseRegressorDistribution):
    """Universal regressor distribution."""

    method: str
    """The resampling method."""

    size: int
    """The size of the resampling set."""

    weights: list[Callable[[NumberArray], float]]
    """The weight functions related to the sub-algorithms.

    A weight function computes a weight from an input data array.
    """

    N_BOOTSTRAP: ClassVar[int] = 100
    """The default number of replicates for the bootstrap method."""

    N_FOLDS: ClassVar[int] = 5
    """The default number of folds for the cross-validation method."""

    CROSS_VALIDATION: Final[str] = "cv"
    BOOTSTRAP: Final[str] = "b"
    LOO: Final[str] = "loo"

    def __init__(
        self,
        algo: BaseRegressor,
        bootstrap: bool = True,
        loo: bool = False,
        size: int | None = None,
    ) -> None:
        """
        Args:
            bootstrap: The resampling method.
                If `True`, use bootstrap resampling.
                Otherwise, use cross-validation resampling.
            loo: The leave-One-Out sub-method, when bootstrap is `False`.
                If `False`, use parameterized cross-validation,
                Otherwise use leave-one-out.
            size: The size of the resampling set,
                i.e. the number of times the regression algorithm is rebuilt.
                If `None`,
                [N_BOOTSTRAP][gemseo_mlearning.active_learning.distributions.regressor_distribution.RegressorDistribution.N_BOOTSTRAP]
                in the case of bootstrap
                and
                [N_FOLDS][gemseo_mlearning.active_learning.distributions.regressor_distribution.RegressorDistribution.N_FOLDS]
                in the case of cross-validation.
                This argument is ignored in the case of leave-one-out.
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

            algo.learn(new_samples.tolist())

    def __weight_function(self, indices: list[int]) -> Callable[[NumberArray], float]:
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
            input_data: NumberArray,
        ) -> float:
            """Weight function returning a weight when a 1D input array is passed.

            Args:
                input_data: A 1D input array.

            Returns:
                A weight related to the input array.
            """
            if isinstance(input_data, Mapping):
                input_data = concatenate_dict_of_arrays_to_array(
                    input_data, self.input_names
                )
            only_one_element = input_data.ndim == 1
            input_data = atleast_2d(input_data)
            distance = ones(input_data.shape[0])
            learning_data = self.learning_set.get_view(group_names=in_grp).to_numpy()
            learning_data = learning_data[indices]
            for learning_datum in learning_data:
                for index, input_datum in enumerate(input_data):
                    distance[index] *= 1 - exp(
                        -(euclidean(learning_datum, input_datum) ** 2) / rho**2
                    )
            if only_one_element:
                distance = distance[0]
            return distance

        return weight

    def predict_members(self, input_data: DataType) -> DataType:
        """Predict the output value with the different machine learning algorithms.

        After prediction, the method stacks the results.

        Args:
            input_data: The input data,
                specified as either as a NumPy array or as dictionary of NumPy arrays
                indexed by inputs names.
                The NumPy array can be either a `(d,)` array
                representing a sample in dimension `d`,
                or a `(M, d)` array representing `M` samples in dimension `d`.

        Returns:
            The output data (dimension `p`) of `N` machine learning algorithms.
                If `input_data.shape == (d,)`, then `output_data.shape == (N, p)`.
                If `input_data.shape == (M,d)`, then `output_data.shape == (N,M,p)`.
        """
        predictions = [algo.predict(input_data) for algo in self.algos]
        if isinstance(input_data, Mapping):
            return {
                name: stack([prediction[name] for prediction in predictions])
                for name in predictions[0]
            }
        return stack(predictions)

    def compute_confidence_interval(  # noqa: D102
        self,
        input_data: DataType,
        level: float = 0.95,
    ) -> (
        tuple[dict[str, NumberArray], dict[str, NumberArray]]
        | tuple[NumberArray, NumberArray]
    ):
        level = (1.0 - level) / 2.0
        predictions = self.predict_members(input_data)
        if isinstance(predictions, Mapping):
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

    def evaluate_weights(self, input_data: NumberArray) -> NumberArray:
        """Evaluate weights.

        Args:
            input_data: The input data with shape (size, n_input_data)

        Returns:
            The weights.
        """
        weights = array([func(input_data) for func in self.weights])
        return weights / npsum(weights, 0)

    @RegressionDataFormatters.format_dict
    @RegressionDataFormatters.format_samples()
    def compute_mean(self, input_data: DataType) -> DataType:  # noqa: D102
        predictions = self.predict_members(input_data)
        weights = self.evaluate_weights(input_data)
        return self.__average(weights, predictions)

    @RegressionDataFormatters.format_dict
    @RegressionDataFormatters.format_samples()
    def compute_variance(self, input_data: DataType) -> DataType:  # noqa: D102
        predictions = self.predict_members(input_data)
        weights = self.evaluate_weights(input_data)
        term1 = self.__average(weights, predictions**2)
        term2 = self.__average(weights, predictions) ** 2
        return term1 - term2

    def compute_samples(  # noqa: D102
        self,
        input_data: NumberArray,
        n_samples: int,
    ) -> NumberArray:
        return self.predict_members(input_data)

    @staticmethod
    def __average(
        weights: NumberArray,
        data: NumberArray,
    ) -> NumberArray:
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

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
"""Mixin for expected improvement."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar
from typing import Literal

from numpy import array
from numpy import atleast_2d
from numpy import dot
from numpy import maximum
from numpy import nan_to_num
from scipy.stats import norm

from gemseo_mlearning.active_learning.distributions.kriging_distribution import (
    KrigingDistribution,
)

if TYPE_CHECKING:
    from gemseo.mlearning.core.algos.ml_algo import DataType
    from gemseo.typing import NumberArray

    from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (  # noqa: E501
        BaseRegressorDistribution,
    )
    from gemseo_mlearning.active_learning.distributions.regressor_distribution import (
        RegressorDistribution,
    )


class ExpectedImprovement:
    """A mixin for expected improvement."""

    _OPTIMIZE: ClassVar[Callable[[NumberArray], float]] = max
    """The optimization function."""

    _SIGN: ClassVar[Literal[-1] | Literal[1]] = 1
    """The sign."""

    __compute_ei: Callable[[DataType, float], DataType]
    """The function computing the expected improvement."""

    __optimum: NumberArray
    """The current optimum estimation."""

    def __init__(  # noqa: D107
        self,
        regressor_distribution: BaseRegressorDistribution,
    ) -> None:
        super().__init__(regressor_distribution)
        if isinstance(regressor_distribution, KrigingDistribution):
            self.__compute_ei = self.__compute_gaussian_expected_improvement
        else:
            self.__compute_ei = self.__compute_empirical_expected_improvement

    def update(self) -> None:
        data = self._regressor_distribution.learning_set.output_dataset.to_numpy()
        self.__optimum = self._OPTIMIZE(data)

    def _compute_output(self, input_value: NumberArray) -> NumberArray:  # noqa: D102
        return self.__compute_ei(input_value) / self._scaling_factor

    def __compute_gaussian_expected_improvement(  # noqa: D102
        self, input_data: DataType
    ) -> DataType:
        """Compute the expected improvement from a Gaussian process regressor.

        Args:
            input_data: The input point at which to compute the expected improvement.

        Returns:
            The expected improvement.
        """
        improvement = (self._compute_mean(input_data) - self.__optimum) * self._SIGN
        std = self._compute_standard_deviation(input_data)
        value = nan_to_num(improvement / std)
        return improvement * norm.cdf(value) + std * norm.pdf(value)

    def __compute_empirical_expected_improvement(  # noqa: D102
        self, input_data: DataType
    ) -> DataType:
        """Compute the expected improvement from a regressor.

        Args:
            input_data: The input point at which to compute the expected improvement.

        Returns:
            The expected improvement.
        """
        self._regressor_distribution: RegressorDistribution
        input_data = atleast_2d(input_data)
        predictions = self._regressor_distribution.predict_members(input_data)
        weights = self._regressor_distribution.evaluate_weights(input_data)
        expected_improvement = maximum((predictions - self.__optimum) * self._SIGN, 0.0)
        return array([
            dot(weights[:, index], expected_improvement[:, index, :])
            for index in range(expected_improvement.shape[1])
        ])

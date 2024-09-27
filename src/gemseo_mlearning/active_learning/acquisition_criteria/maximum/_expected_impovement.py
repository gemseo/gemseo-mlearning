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
from numpy import max as np_max
from numpy import maximum
from numpy import mean
from numpy import nan_to_num
from scipy.stats import norm

if TYPE_CHECKING:
    from gemseo.typing import NumberArray

    from gemseo_mlearning.active_learning.distributions.regressor_distribution import (
        RegressorDistribution,
    )


class ExpectedImprovement:
    """A mixin for expected improvement."""

    _OPTIMIZE: ClassVar[Callable[[NumberArray], float]] = max
    """The optimization function."""

    _SIGN: ClassVar[Literal[-1, 1]] = 1
    """The sign."""

    def update(self) -> None:
        self._qoi = self._OPTIMIZE(
            self._regressor_distribution.learning_set.output_dataset.to_numpy()
        )

    def _compute(  # noqa: D102
        self, input_data: NumberArray
    ) -> NumberArray | float:
        # See Proposition 14, Jones et al, 1998
        improvement = (self._compute_mean(input_data) - self._qoi) * self._SIGN
        std = self._compute_standard_deviation(input_data)
        value = nan_to_num(improvement / std)
        return (
            improvement * norm.cdf(value) + std * norm.pdf(value)
        ) / self._scaling_factor

    def _compute_empirically(  # noqa: D102
        self, input_data: NumberArray
    ) -> NumberArray | float:
        self._regressor_distribution: RegressorDistribution
        ndim_is_two = input_data.ndim == 2
        input_data = atleast_2d(input_data)
        predictions = self._regressor_distribution.predict_members(input_data)
        weights = self._regressor_distribution.evaluate_weights(input_data)
        expected_improvement = maximum((predictions - self._qoi) * self._SIGN, 0.0)
        expected_improvement = array([
            dot(weights[:, index], expected_improvement[:, index, :])
            for index in range(expected_improvement.shape[1])
        ])
        if ndim_is_two:
            return expected_improvement / self._scaling_factor

        return expected_improvement[0] / self._scaling_factor

    def _compute_by_batch(  # noqa: D102
        self, q_input_values: NumberArray
    ) -> NumberArray | float:
        q_input_values = self._reshape_input_values(q_input_values)
        try:
            samples = self._compute_samples(
                input_data=q_input_values, n_samples=self._mc_size
            )
            improvement = (samples - self._qoi) * self._SIGN
            return mean(np_max(maximum(improvement, 0), axis=0)) / self._scaling_factor
        # distribution is not positive definite.
        except TypeError:
            return 0.0

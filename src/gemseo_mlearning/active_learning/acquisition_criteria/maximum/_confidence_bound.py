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
"""Mixin for confidence bounds."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import abs as np_abs
from numpy import inf
from numpy import max as np_max
from numpy import mean
from numpy import nan_to_num
from numpy import pi
from numpy import sign
from numpy import sqrt

if TYPE_CHECKING:
    from gemseo.typing import NumberArray

    from gemseo_mlearning.active_learning.distributions import BaseRegressorDistribution


class ConfidenceBound:
    """A mixin for confidence bounds."""

    __kappa: float
    """The factor associated with the standard deviation."""

    def __init__(
        self,
        regressor_distribution: BaseRegressorDistribution,
        kappa: float = 2.0,
        batch_size: int = 1,
        mc_size: int = 10_000,
    ) -> None:
        """
        Args:
            regressor_distribution: The distribution of the regressor.
            kappa: A factor associated with the standard deviation
                to increase the mean value.
            batch_size: The number of points to be acquired in parallel;
                if `1`, acquire points sequentially.
            mc_size: The sample size to estimate the acquisition criterion in parallel.
        """  # noqa: D205 D212 D415
        self.__kappa = kappa
        super().__init__(regressor_distribution, batch_size=batch_size, mc_size=mc_size)

    def _compute(self, input_value: NumberArray) -> NumberArray | float:  # noqa: D102
        mean = self._compute_mean(input_value)
        sigma = self._compute_standard_deviation(input_value)
        return (mean + self.__kappa * sigma) / self._scaling_factor

    def _compute_by_batch(  # noqa: D102
        self, q_input_values: NumberArray
    ) -> NumberArray | float:
        # See Equation 8 in
        # Wilson, J. T., Moriconi, R., Hutter, F., & Deisenroth, M. P. (2017).
        # The reparameterization trick for acquisition functions.
        # arXiv preprint arXiv:1712.00424.
        q_input_values = self._reshape_input_values(q_input_values)
        kappa_sign = sign(self.__kappa)
        try:
            samples = self._compute_samples(
                input_data=q_input_values, n_samples=self._mc_size
            )[..., 0]
        except TypeError:
            # The covariance matrix is not positive definite.
            return -kappa_sign * nan_to_num(inf)

        means = self._compute_mean(q_input_values).ravel()
        centered_samples = samples - means
        return (
            mean(
                kappa_sign
                * np_max(
                    kappa_sign
                    * (means + self.__kappa * sqrt(pi / 2) * np_abs(centered_samples)),
                    axis=0,
                )
            )
            / self._scaling_factor
        )

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
"""Expected feasibility."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from numpy import array
from numpy import atleast_2d
from numpy import dot
from numpy import max as np_max
from numpy import maximum
from numpy import mean
from numpy import nan_to_num
from scipy.stats import norm

from gemseo_mlearning.active_learning.acquisition_criteria.level_set.base_ei_ef import (  # noqa: E501
    BaseEIEF,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray

    from gemseo_mlearning.active_learning.distributions.regressor_distribution import (  # noqa: E501
        RegressorDistribution,
    )

warnings.filterwarnings("ignore")


class EF(BaseEIEF):
    r"""The expected feasibility.

    This acquisition criterion is expressed as

    $$EF[x]=\mathbb{E}\left[\max\left(\kappa\mathbb{S}[Y(x)]-|y-Y(x)|,0\right)\right]$$

    where $y$ is the model output value characterizing the level set
    and $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$.

    In the case of a Gaussian process regressor,
    it has an analytic expression:

    $$
    EF[x] =
    \mathbb{S}[Y(x)]
    (
    \kappa
    (\Phi(t^+)-\Phi(t^-))
    -t(2\Phi(t)-\Phi(t^+)-\Phi(t^-))
    -(2\phi(t)-\phi(t^+)-\phi(t^-))
    )
    $$

    where $\Phi$ and $\phi$ are respectively
    the cumulative and probability density functions
    of the standard normal distribution,
    $t=\frac{y - \mathbb{E}[Y(x)]}{\mathbb{S}[Y(x)]}$,
    $t^+=t+\kappa$,
    $t^-=t-\kappa$
    and $\kappa>0$.

    For the acquisition of $q>1$ points at a time,
    the acquisition criterion changes to

    $$EF[x_1,\dots,x_q] = \mathbb{E}\left[\max_{1\leq i \leq q}\left(
    \max(\kappa\mathbb{S}[Y(x_i)] - |y - Y(x_i)|,0)
    \right)\right]$$

    where the expectation is taken with respect to the distribution of
    the random vector $(Y(x_1),\dots,Y(x_q))$.
    There is no analytic expression
    and the acquisition is thus instead evaluated with crude Monte-Carlo.
    """

    def _compute(self, input_value: NumberArray) -> NumberArray | float:  # noqa: D102
        # See Proposition 4, Bect et al, 2012
        standard_deviation, t, t_m, t_p = self._get_material(input_value)
        # As the next line would log warnings when value is a big number,
        # we filter these warnings before defining the class
        return nan_to_num(
            standard_deviation
            * (
                self._kappa * ((cdf_t_p := norm.cdf(t_p)) - (cdf_t_m := norm.cdf(t_m)))
                - t * (2 * norm.cdf(t) - cdf_t_p - cdf_t_m)
                - (2 * norm.pdf(t) - norm.pdf(t_p) - norm.pdf(t_m))
            ),
        )

    def _compute_empirically(  # noqa: D102
        self, input_value: NumberArray
    ) -> NumberArray | float:
        self._regressor_distribution: RegressorDistribution
        ndim_is_two = input_value.ndim == 2
        input_data = atleast_2d(input_value)
        samples = self._compute_samples(
            input_data=input_data, n_samples=self._compute_samples
        )
        weights = self._regressor_distribution.evaluate_weights(input_data)
        feasibility = self._kappa * self._compute_standard_deviation(input_value) - abs(
            self._output_value - samples
        )
        expected_feasibility = maximum(feasibility, 0.0)
        expected_feasibility = array([
            dot(weights[:, index], expected_feasibility[:, index, :])
            for index in range(expected_feasibility.shape[1])
        ])
        if ndim_is_two:
            return nan_to_num(expected_feasibility)

        return nan_to_num(expected_feasibility[0])

    def _compute_by_batch(self, q_input_values: NumberArray) -> NumberArray | float:  # noqa: D102
        q_input_values = self._reshape_input_values(q_input_values)
        try:
            samples = self._compute_samples(
                input_data=q_input_values, n_samples=self._mc_size
            )[..., 0]
        except TypeError:
            # The covariance matrix is not positive definite.
            return 0.0

        return nan_to_num(
            mean(
                np_max(
                    maximum(
                        self._kappa * self._compute_standard_deviation(q_input_values)
                        - abs(self._output_value - samples.T),
                        0,
                    ),
                    axis=1,
                ),
            ),
        )

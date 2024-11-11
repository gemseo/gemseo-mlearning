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
"""Expected improvement."""

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
from numpy import square
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


class EI(BaseEIEF):
    r"""The expected improvement.

    This acquisition criterion is expressed as

    $$EI[x]
    =\mathbb{E}\left[\max\left((\kappa\mathbb{S}[Y(x)])^2-(y - Y(x))^2,0\right)\right]$$

    where $y$ is the model output value characterizing the level set
    and $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$.

    In the case of a Gaussian process regressor,
    it has an analytic expression:

    $$
    EI[x] = \mathbb{V}[Y(x)]\times
    (
    (\kappa^2-1-t^2)(\Phi(t^+)-\Phi(t^-))
    -2t(\phi(t^+)-\phi(t^-))
    +t^+\phi(t^+)
    -t^-\phi(t^-)
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

    $$EI[x_1,\dots,x_q] = \mathbb{E}\left[\max_{1\leq i \leq q}\left(
    \max((\kappa\mathbb{S}[Y(x_i)])^2 - (y - Y(x_i))^2,0)
    \right)\right]$$

    where the expectation is taken with respect to the distribution of
    the random vector $(Y(x_1),\dots,Y(x_q))$.
    There is no analytic expression
    and the acquisition is thus instead evaluated with crude Monte-Carlo.
    """

    def _compute(self, input_value: NumberArray) -> NumberArray | float:  # noqa: D102
        # See Proposition 4, Bect et al, 2012
        standard_deviation, t, t_m, t_p = self._get_material(input_value)
        return nan_to_num(
            standard_deviation**2
            * (
                (self._kappa**2 - 1 - t**2) * (norm.cdf(t_p) - norm.cdf(t_m))
                - 2 * t * ((pdf_t_p := norm.pdf(t_p)) - (pdf_t_m := norm.pdf(t_m)))
                + t_p * pdf_t_p
                - t_m * pdf_t_m
            ),
        )

    def _compute_empirically(  # noqa: D102
        self, input_value: NumberArray
    ) -> NumberArray | float:
        # See Proposition 14, Ben Salem et al, 1998
        self._regressor_distribution: RegressorDistribution
        ndim_is_two = input_value.ndim == 2
        input_data = atleast_2d(input_value)
        samples = self._compute_samples(
            input_data=input_data, n_samples=self._compute_samples
        )
        weights = self._regressor_distribution.evaluate_weights(input_data)
        improvement = (
            self._kappa * self._compute_standard_deviation(input_value)
        ) ** 2 - square(self._output_value - samples)
        expected_improvement = maximum(improvement, 0.0)
        expected_improvement = array([
            dot(weights[:, index], expected_improvement[:, index, :])
            for index in range(expected_improvement.shape[1])
        ])
        if ndim_is_two:
            return nan_to_num(expected_improvement)

        return nan_to_num(expected_improvement[0])

    def _compute_by_batch(self, q_input_values: NumberArray) -> NumberArray | float:  # noqa: D102
        q_input_values = self._reshape_input_values(q_input_values)
        try:
            samples = self._compute_samples(
                input_data=q_input_values, n_samples=self._mc_size
            )[..., 0]
            return nan_to_num(
                mean(
                    np_max(
                        maximum(
                            (
                                self._kappa
                                * self._compute_standard_deviation(q_input_values)
                            )
                            ** 2
                            - square(self._output_value - samples.T),
                            0,
                        ),
                        axis=1,
                    ),
                ),
            )
        # distribution is not positive definite.
        except TypeError:
            return 0.0

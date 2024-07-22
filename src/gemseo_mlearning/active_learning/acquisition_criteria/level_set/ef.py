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

from typing import TYPE_CHECKING

from scipy.stats import norm

from gemseo_mlearning.active_learning.acquisition_criteria.level_set.base_ei_ef import (  # noqa: E501
    BaseEIEF,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class EF(BaseEIEF):
    r"""The expected feasibility.

    This acquisition criterion is expressed as:

    $$
    EF[x] = \mathbb{S}[Y(x)]\times
    (
    \kappa
    (\Phi(t^+)-\Phi(t^-))
    -t(2\Phi(t)-\Phi(t^+)-\Phi(t^-))
    -(2\phi(t)-\phi(t^+)-\phi(t^-))
    )
    $$

    where $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$,
    $t=\frac{y - \mathbb{E}[Y(x)]}{\mathbb{E}[Y(x)]}$,
    $t^+=t+\kappa$,
    $t^-=t-\kappa$,
    $y$ is the model output value characterizing the level set
    and $\kappa\in[0,1]$ (default: 1).
    """

    def _compute_output(self, input_value: NumberArray) -> NumberArray:  # noqa: D102
        # See Proposition 4, Bect et al, 2012
        standard_deviation, t, t_minus, t_plus = self._get_material(input_value)
        cdf_t_plus = norm.cdf(t_plus)
        cdf_t_minus = norm.cdf(t_minus)
        return standard_deviation * (
            self._kappa * (cdf_t_plus - cdf_t_minus)
            - t * (2 * norm.cdf(t) - cdf_t_plus - cdf_t_minus)
            - (2 * norm.pdf(t) - norm.pdf(t_plus) - norm.pdf(t_minus))
        )

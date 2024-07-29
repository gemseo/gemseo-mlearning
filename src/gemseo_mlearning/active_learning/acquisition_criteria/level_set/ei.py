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

from typing import TYPE_CHECKING

from scipy.stats import norm

from gemseo_mlearning.active_learning.acquisition_criteria.level_set.base_ei_ef import (  # noqa: E501
    BaseEIEF,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class EI(BaseEIEF):
    r"""The expected improvement.

    This acquisition criterion is expressed as:

    $$
    EI[x] = \mathbb{V}[Y(x)]\times
    (
    (\kappa^2-1-t^2)(\Phi(t^+)-\Phi(t^-))
    -2t(\phi(t^+)-\phi(t^-))
    +t^+\phi(t^+)
    -t^-\phi(t^-)
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
        standard_deviation, t, t_m, t_p = self._get_material(input_value)
        return standard_deviation**2 * (
            (self._kappa**2 - 1 - t**2) * (norm.cdf(t_p) - norm.cdf(t_m))
            - 2 * t * ((pdf_t_p := norm.pdf(t_p)) - (pdf_t_m := norm.pdf(t_m)))
            + t_p * pdf_t_p
            - t_m * pdf_t_m
        )

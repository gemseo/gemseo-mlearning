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
"""U-function."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import inf
from numpy import mean
from numpy import min as np_min
from numpy import nan_to_num
from numpy import square

from gemseo_mlearning.active_learning.acquisition_criteria.level_set.base_level_set import (  # noqa: E501
    BaseLevelSet,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class U(BaseLevelSet):
    r"""The U-function.

    This acquisition criterion is expressed as

    $$U[x] = \mathbb{E}\left[\left(\frac{y-Y(x)}{\mathbb{S}[Y(x)]}
    \right)^2\right]$$

    where $y$ is the model output value characterizing the level set
    and $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$.
    It has an analytic expression:

    $$U[x] = \left(\frac{y-\mathbb{E}[Y(x)]}{\mathbb{S}[Y(x)]}\right)^2$$

    where $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$
    and $y$ is the model output value characterizing the level set.
    For numerical purposes,
    the expression effectively minimized corresponds to its square root.

    For the acquisition of $q>1$ points at a time,
    the acquisition criterion changes to

    $$U[x_1,\dots,x_q] = \mathbb{E}\left[\min\left(
    \left(\frac{y-Y(x_1)}{\mathbb{S}[Y(x_1)]}\right)^2,\dots,
    \left(\frac{y-Y(x_q)}{\mathbb{S}[Y(x_q)]}
    \right)^2\right)\right]$$

    where the expectation is taken with respect to the distribution of
    the random vector $(Y(x_1),\dots,Y(x_q))$.
    There is no analytic expression
    and the acquisition is thus instead evaluated with crude Monte-Carlo.
    """

    MAXIMIZE: ClassVar[bool] = False

    def _compute(self, input_value: NumberArray) -> NumberArray | float:  # noqa: D102
        # See Proposition 12, Roy \& Notz, 2014
        return nan_to_num(
            abs(self._output_value - self._compute_mean(input_value))
            / self._compute_standard_deviation(input_value),
            nan=nan_to_num(inf),
        )

    def _compute_by_batch(self, q_input_values: NumberArray) -> NumberArray | float:  # noqa: D102
        q_input_values = self._reshape_input_values(q_input_values)
        try:
            samples = self._compute_samples(
                input_data=q_input_values, n_samples=self._mc_size
            )
            return nan_to_num(
                mean(
                    np_min(
                        square(self._output_value - samples)
                        / self._compute_variance(q_input_values),
                        axis=1,
                    )
                ),
                nan=nan_to_num(inf),
            )
        # distribution is not positive definite.
        except TypeError:
            return nan_to_num(inf)

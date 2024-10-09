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
from typing import Callable
from typing import ClassVar
from typing import Literal

from gemseo_mlearning.active_learning.acquisition_criteria.maximum._expected_impovement import (  # noqa: E501
    ExpectedImprovement,
)
from gemseo_mlearning.active_learning.acquisition_criteria.minimum.base_minimum import (
    BaseMinimum,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class EI(ExpectedImprovement, BaseMinimum):
    r"""Expected improvement.

    This acquisition criterion is expressed as

    $$EI[x] = \mathbb{E}[\max(y_{\text{min}}-Y(x),0)]$$

    where $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$
    and $y_{\text{min}}$ is the minimum output value in the learning set.

    In the case of a Gaussian regressor,
    it has an analytic expression:

    $$
    EI[x] = (y_{\text{min}}-\mathbb{E}[Y(x)])\Phi(t)
    + \mathbb{S}[Y(x)]\phi(t)
    $$

    where $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$,
    $t=\frac{y_{\text{min}}-\mathbb{E}[Y(x)]}{\mathbb{S}[Y(x)]}$.

    For the acquisition of $q>1$ points at a time,
    the acquisition criterion changes to

    $$EI[x_1,\dots,x_q] = \mathbb{E}\left[\max\left(
    \max(y_{\text{min}}-Y(x_1),0),\dots,
    \max(y_{\text{min}}-Y(x_q),0)
    \right)\right]$$

    where the expectation is taken with respect to the distribution of
    the random vector $(Y(x_1),\dots,Y(x_q))$.
    There is no analytic expression
    and the acquisition is thus instead evaluated with crude Monte-Carlo.
    """

    _OPTIMIZE: ClassVar[Callable[[NumberArray], float]] = min
    """The optimization function."""

    _SIGN: ClassVar[Literal[-1, 1]] = -1
    """The sign."""

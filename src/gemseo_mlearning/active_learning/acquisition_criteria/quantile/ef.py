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

from gemseo_mlearning.active_learning.acquisition_criteria.level_set.ef import EF as _EF
from gemseo_mlearning.active_learning.acquisition_criteria.quantile.base_quantile import (  # noqa: E501
    BaseQuantile,
)


class EF(BaseQuantile):
    r"""The expected feasibility.

    This acquisition criterion is expressed as

    $$
    EF[x] = \mathbb{E}\left[\max(\kappa\mathbb{S}[Y(x)] - |y_{\alpha} - Y(x)|,0)\right]
    $$

    where $y_{\alpha}$ is the $\alpha$-quantile of the model output
    and $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$.

    In the case of a Gaussian process regressor,
    it has an analytic expression:

    $$
    EF[x] = \mathbb{S}[Y(x)]\times
    \left(
    \kappa
    (\Phi(t^+)-\Phi(t^-))
    -t(2\Phi(t)-\Phi(t^+)-\Phi(t^-))
    -(2\phi(t)-\phi(t^+)-\phi(t^-))
    \right)
    $$

    where $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$,
    $t=\frac{y_{\alpha} - \mathbb{E}[Y(x)]}{\mathbb{S}[Y(x)]}$,
    $t^+=t+\kappa$,
    $t^-=t-\kappa$
    and $\kappa>0$.

    For the acquisition of $q>1$ points at a time,
    the acquisition criterion changes to

    $$EF[x_1,\dots,x_q] = \mathbb{E}\left[\max_{1\leq i\leq q}\left(
    \max(\kappa\mathbb{S}[Y(x_i)] - |y_{\alpha} - Y(x_i)|,0)
    \right)\right]$$

    where the expectation is taken with respect to the distribution of
    the random vector $(Y(x_1),\dots,Y(x_q))$.
    There is no analytic expression
    and the acquisition is thus instead evaluated with crude Monte-Carlo.
    """

    _LEVEL_SET_CLASS = _EF

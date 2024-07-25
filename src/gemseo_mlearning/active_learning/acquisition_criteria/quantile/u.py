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

from typing import ClassVar

from gemseo_mlearning.active_learning.acquisition_criteria.level_set.u import U as _U
from gemseo_mlearning.active_learning.acquisition_criteria.quantile.base_quantile import (  # noqa: E501
    BaseQuantile,
)


class U(BaseQuantile):
    r"""The U-function.

    This acquisition criterion is expressed as:

    $$U[x] = \frac{|y_{\alpha}-\mathbb{E}[Y(x)]|}{\mathbb{S}[Y(x)])}$$

    where $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$
    and $y_{\alpha}$ is the $\alpha$-quantile of the model output.
    """

    MAXIMIZE: ClassVar[bool] = False

    _LEVEL_SET_CLASS = _U
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

from gemseo_mlearning.active_learning.acquisition_criteria.maximum._expected_impovement import (  # noqa: E501
    ExpectedImprovement,
)
from gemseo_mlearning.active_learning.acquisition_criteria.maximum.base_maximum import (
    BaseMaximum,
)


class EI(ExpectedImprovement, BaseMaximum):
    r"""Expected improvement.

    This acquisition criterion is expressed as:

    $$EI[x] = \mathbb{E}[\max(Y(x)-y_{\text{max}},0)]$$

    where $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$
    and $y_{\text{max}}$ is the maximum output values in the learning set.
    """

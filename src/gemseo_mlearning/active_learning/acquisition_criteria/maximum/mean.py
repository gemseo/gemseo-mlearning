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
"""Mean-based criterion."""

from __future__ import annotations

from gemseo_mlearning.active_learning.acquisition_criteria.maximum._mean import (
    Mean as _Mean,
)
from gemseo_mlearning.active_learning.acquisition_criteria.maximum.base_maximum import (
    BaseMaximum,
)


class Mean(_Mean, BaseMaximum):
    r"""Mean-based criterion.

    This acquisition criterion is expressed as

    $$E[x] = \mathbb{E}[Y(x)]$$

    where $Y$ is the random process modelling the uncertainty of the surrogate model
    $\hat{f}$.
    """

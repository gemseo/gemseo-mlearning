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
"""Upper confidence bound (UCB)."""

from __future__ import annotations

from gemseo_mlearning.active_learning.acquisition_criteria.maximum._confidence_bound import (  # noqa: E501
    ConfidenceBound,
)
from gemseo_mlearning.active_learning.acquisition_criteria.maximum.base_maximum import (
    BaseMaximum,
)


class UCB(ConfidenceBound, BaseMaximum):
    r"""The upper confidence bound (UCB).

    This acquisition criterion is expressed as:

    $$M[x;\kappa] = \mathbb{E}[Y(x)] + \kappa \times \mathbb{S}[Y(x)]$$

    where $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$
    and $\kappa>0$.
    """

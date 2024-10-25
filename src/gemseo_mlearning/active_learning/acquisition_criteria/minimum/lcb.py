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
"""Lower confidence bound (LCB)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo_mlearning.active_learning.acquisition_criteria.maximum._confidence_bound import (  # noqa: E501
    ConfidenceBound,
)
from gemseo_mlearning.active_learning.acquisition_criteria.minimum.base_minimum import (
    BaseMinimum,
)

if TYPE_CHECKING:
    from gemseo_mlearning.active_learning.distributions import BaseRegressorDistribution


class LCB(ConfidenceBound, BaseMinimum):
    r"""The lower confidence bound (LCB).

    This acquisition criterion is expressed as

    $$M[x;\kappa] = \mathbb{E}[Y(x)] - \kappa \times \mathbb{S}[Y(x)]$$

    where $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$
    and $\kappa>0$.
    """

    MAXIMIZE: ClassVar[bool] = False

    def __init__(  # noqa: D107
        self,
        regressor_distribution: BaseRegressorDistribution,
        kappa: float = 2.0,
        batch_size: int = 1,
        mc_size: int = 10_000,
    ) -> None:  # noqa: D102
        super().__init__(
            regressor_distribution,
            kappa=-abs(kappa),
            batch_size=batch_size,
            mc_size=mc_size,
        )

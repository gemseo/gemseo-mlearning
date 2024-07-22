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

    This acquisition criterion is expressed as:

    $$EI[x] = \mathbb{E}[\max(y_{\text{min}}-Y(x),0)]$$

    where $Y$ is the random process
    modelling the uncertainty of the surrogate model $\hat{f}$
    and $y_{\text{min}}$ is the minimum output values in the learning set.
    """

    _OPTIMIZE: ClassVar[Callable[[NumberArray], float]] = min
    """The optimization function."""

    _SIGN: ClassVar[Literal[-1] | Literal[1]] = -1
    """The sign."""

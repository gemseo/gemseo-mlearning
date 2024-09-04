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
"""Base class for EI and EF criteria to approximate a level set."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_mlearning.active_learning.acquisition_criteria.level_set.base_level_set import (  # noqa: E501
    BaseLevelSet,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray

    from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (  # noqa: E501
        BaseRegressorDistribution,
    )


class BaseEIEF(BaseLevelSet):
    """The base class for EI and EF criteria to approximate a level set.

    EI and EF stands for expected improvement and expected feasibility respectively.
    More information in the subclasses
    [EI][gemseo_mlearning.active_learning.acquisition_criteria.level_set.ei.EI]
    and
    [EF][gemseo_mlearning.active_learning.acquisition_criteria.level_set.ef.EF].
    """

    _kappa: float
    """A percentage of the standard deviation around `output_value` to add points."""

    def __init__(
        self,
        regressor_distribution: BaseRegressorDistribution,
        output_value: float,
        kappa: float = 2.0,
    ) -> None:
        """
        Args:
            kappa: A percentage of the standard deviation
                describing the area around `output_value` where to add points.
        """  # noqa: D205 D212 D415
        super().__init__(regressor_distribution, output_value)
        self._kappa = kappa

    def _get_material(
        self, input_data: NumberArray
    ) -> tuple[NumberArray, NumberArray, NumberArray, NumberArray]:
        """Compute material for EI and EF.

        Args:
            input_data: The input value(s) of the model.

        Returns:
            The standard deviation of the output
            and three quantities of [Bect et al, 2012, Proposition 4:
            `t`, `t_minus` and `t_plus`.
        """
        mean = self._compute_mean(input_data)
        standard_deviation = self._compute_standard_deviation(input_data)
        t = (self._output_value - mean) / standard_deviation
        t_plus = t + self._kappa
        t_minus = t - self._kappa
        return standard_deviation, t, t_minus, t_plus

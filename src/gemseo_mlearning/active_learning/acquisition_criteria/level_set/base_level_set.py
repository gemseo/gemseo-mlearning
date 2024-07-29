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
#
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base class for acquisition criteria to estimate a level set."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion import (  # noqa: E501
    BaseAcquisitionCriterion,
)

if TYPE_CHECKING:
    from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (  # noqa: E501
        BaseRegressorDistribution,
    )


class BaseLevelSet(BaseAcquisitionCriterion):
    """The base class for acquisition criteria to estimate a level set estimation."""

    _output_value: float
    """The model output value characterizing the level set."""

    def __init__(
        self,
        regressor_distribution: BaseRegressorDistribution,
        output_value: float,
    ) -> None:
        """
        Args:
            output_value: The model output value characterizing the level set.
        """  # noqa: D205 D212 D415
        super().__init__(regressor_distribution)
        self._output_value = output_value

    def update(self, output_value: float | None = None) -> None:
        """
        Args:
            output_value: The model output value characterizing the level set.
                If ``None``, do not update the acquisition criterion.
        """  # noqa: D205, D212
        super().update()
        if output_value is not None:
            self._output_value = output_value

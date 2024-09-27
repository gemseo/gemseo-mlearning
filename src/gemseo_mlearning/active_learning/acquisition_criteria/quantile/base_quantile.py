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
"""Base class for acquisition criteria to estimate a quantile."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo.utils.string_tools import pretty_str
from numpy import atleast_1d
from numpy import quantile

from gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion import (  # noqa: E501
    BaseAcquisitionCriterion,
)

if TYPE_CHECKING:
    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.typing import NumberArray
    from gemseo.typing import RealArray

    from gemseo_mlearning.active_learning.acquisition_criteria.level_set.base_level_set import (  # noqa: E501
        BaseLevelSet,
    )
    from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (  # noqa: E501
        BaseRegressorDistribution,
    )


class BaseQuantile(BaseAcquisitionCriterion):
    """The base class for acquisition criteria to estimate a quantile."""

    @property
    @abstractmethod
    def _LEVEL_SET_CLASS(self) -> type[BaseLevelSet]:  # noqa: N802
        """The acquisition criterion to estimate a level set."""

    __level: float
    """The quantile level."""

    __input_data: RealArray
    """The input samples to estimate the quantile."""

    def __init__(
        self,
        regressor_distribution: BaseRegressorDistribution,
        level: float,
        uncertain_space: ParameterSpace,
        n_samples: int = 100000,
        batch_size: int = 1,
        mc_size: int = 10000,
    ) -> None:
        """
        Args:
            level: The quantile level.
            uncertain_space: The uncertain variable space.
            n_samples: The number of samples
                to estimate the quantile of the regressor by Monte Carlo.
        """  # noqa: D205 D212 D415
        input_names = regressor_distribution.input_names
        missing_names = set(input_names) - set(uncertain_space.variable_names)
        if missing_names:
            msg = (
                "The probability distributions of the input variables "
                f"{pretty_str(missing_names, use_and=True)} are missing."
            )
            raise ValueError(msg)

        # Create a new uncertain space sorted by model inputs.
        self.__uncertain_space = uncertain_space.__class__()
        self.__uncertain_space.add_variables_from(uncertain_space, *input_names)
        self.__input_data = self.__uncertain_space.compute_samples(n_samples)
        self.__level = level
        # The value 0. will be replaced by the quantile estimation at each update,
        # including the first one due to super().__init__.
        self.__level_set_criterion = self._LEVEL_SET_CLASS(
            regressor_distribution, 0.0, batch_size=batch_size, mc_size=mc_size
        )

        super().__init__(regressor_distribution, batch_size=batch_size, mc_size=mc_size)

    def update(self) -> None:  # noqa: D102
        super().update()
        qoi = self._qoi = atleast_1d(self.__compute_quantile())
        self.__level_set_criterion.update(output_value=qoi[0])

    def __compute_quantile(self) -> float:
        """Return the quantile estimation.

        Returns:
            The quantile estimation.
        """
        output_data = self._regressor_distribution.predict(self.__input_data)
        return quantile(output_data, self.__level)

    def _compute(self, input_value: NumberArray) -> NumberArray:
        return self.__level_set_criterion.func(input_value)

    def _compute_empirically(self, input_value: NumberArray) -> NumberArray:
        return self.__level_set_criterion.func(input_value)

    def _compute_by_batch(self, input_value: NumberArray) -> NumberArray:
        return self.__level_set_criterion.func(input_value)

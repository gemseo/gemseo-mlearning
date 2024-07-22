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
"""Base class for acquisition criteria."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from numpy import ones

if TYPE_CHECKING:
    from gemseo.mlearning.core.algos.ml_algo import DataType
    from gemseo.typing import NumberArray

    from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (  # noqa: E501
        BaseRegressorDistribution,
    )


class BaseAcquisitionCriterion(MDOFunction):
    """Base class for acquisition criteria."""

    output_range: float
    """The output range."""

    _compute_mean: Callable[[DataType], DataType]
    """The function to compute the output mean at a given input point."""

    _compute_standard_deviation: Callable[[DataType], DataType]
    """The function to compute the output standard deviation at a given input point."""

    _compute_variance: Callable[[DataType], DataType]
    """The function to compute the output variance at a given input point."""

    _maximize: bool
    """Whether this acquisition criterion must be maximized."""

    _regressor_distribution: BaseRegressorDistribution
    """The distribution of the regressor."""

    MAXIMIZE: ClassVar[bool] = True
    """Whether this acquisition criterion must be maximized."""

    def __init__(
        self,
        regressor_distribution: BaseRegressorDistribution,
    ) -> None:
        """
        Args:
            regressor_distribution: The distribution of the regressor.
        """  # noqa: D205 D212 D415
        self._maximize = True
        self._compute_mean = regressor_distribution.compute_mean
        self._compute_standard_deviation = (
            regressor_distribution.compute_standard_deviation
        )
        self._compute_variance = regressor_distribution.compute_variance
        output_data = regressor_distribution.learning_set.output_dataset.to_numpy()
        self.output_range = output_data.max() - output_data.min()
        self._regressor_distribution = regressor_distribution
        try:
            jac = self._compute_jacobian(
                ones(regressor_distribution.algo.input_dimension)
            )
        except NotImplementedError:
            jac = None
        super().__init__(
            self._compute_output,
            self.__class__.__name__,
            jac=jac,
        )

    @abstractmethod
    def _compute_output(self, input_value: NumberArray) -> NumberArray:
        """Compute the acquisition criterion value.

        Args:
            input_value: The model input value.

        Returns:
            The acquisition criterion value.
        """

    def _compute_jacobian(self, input_value: NumberArray) -> NumberArray:
        """Compute the Jacobian of the acquisition criterion.

        Args:
            input_value: The model input value.

        Returns:
            The Jacobian of the acquisition criterion.

        Raises:
            NotImplementedError: When the function is not implemented.
        """
        raise NotImplementedError

    @property
    def _scaling_factor(self) -> float:
        """The factor to scale values in the output space."""
        if self.output_range == 0:
            return 1.0

        return self.output_range

    def __truediv__(self, other: MDOFunction | float) -> MDOFunction:
        new_criterion = super().__truediv__(other)
        new_criterion._regressor_distribution = self._regressor_distribution
        new_criterion.output_range = self.output_range
        return new_criterion

    def __neg__(self) -> MDOFunction:
        new_criterion = super().__neg__()
        new_criterion._regressor_distribution = self._regressor_distribution
        new_criterion.output_range = self.output_range
        return new_criterion

    def __add__(self, other: MDOFunction | float) -> MDOFunction:
        new_criterion = super().__add__(other)
        new_criterion._regressor_distribution = self._regressor_distribution
        new_criterion.output_range = self.output_range
        return new_criterion

    def __sub__(self, other: MDOFunction | float) -> MDOFunction:
        new_criterion = super().__sub__(other)
        new_criterion._regressor_distribution = self._regressor_distribution
        new_criterion.output_range = self.output_range
        return new_criterion

    def __mul__(self, other: MDOFunction | float) -> MDOFunction:
        new_criterion = super().__mul__(other)
        new_criterion._regressor_distribution = self._regressor_distribution
        new_criterion.output_range = self.output_range
        return new_criterion

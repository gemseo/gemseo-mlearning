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
from typing import Any
from typing import Callable
from typing import ClassVar

from gemseo.core.mdo_functions.mdo_function import MDOFunction
from numpy import ones

from gemseo_mlearning.active_learning.distributions.kriging_distribution import (  # noqa: E501
    KrigingDistribution,
)

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

    _qoi: Any
    """The quantity of interest."""

    _regressor_distribution: BaseRegressorDistribution
    """The distribution of the regressor."""

    _batch_size: int
    """The number of points to be acquired in parallel; if `1`, acquire points
    sequentially."""

    _mc_size: int
    """The sample size to estimate the acquisition criteria in parallel."""

    MAXIMIZE: ClassVar[bool] = True
    """Whether this acquisition criterion must be maximized."""

    def __init__(
        self,
        regressor_distribution: BaseRegressorDistribution,
        batch_size: int = 1,
        mc_size: int = 10_000,
    ) -> None:
        """
        Args:
            regressor_distribution: The distribution of the regressor.
            batch_size: The number of points to be acquired in parallel;
                if `1`, acquire points sequentially.
            mc_size: The sample size to estimate the acquisition criterion in parallel.
        """  # noqa: D205 D212 D415
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

        self._batch_size = batch_size
        self._mc_size = mc_size
        self._compute_samples = regressor_distribution.compute_samples

        if batch_size == 1:
            if isinstance(regressor_distribution, KrigingDistribution):
                compute_output = self._compute
            else:
                compute_output = self._compute_empirically
        else:
            if isinstance(regressor_distribution, KrigingDistribution):
                compute_output = self._compute_by_batch
            else:
                compute_output = self._compute_by_batch_empirically

        self._qoi = None
        super().__init__(
            compute_output,
            self.__class__.__name__,
            jac=jac,
        )

        self.update()

    @property
    def qoi(self) -> Any:
        """The quantity of interest."""
        return self._qoi

    @abstractmethod
    def _compute(self, input_value: NumberArray) -> NumberArray | float:
        """Compute the acquisition criterion value using a random process regressor.

        Args:
            input_value: The model input value.

        Returns:
            The acquisition criterion value.
        """

    def _compute_empirically(self, input_value: NumberArray) -> NumberArray | float:
        """Compute the acquisition criterion value using resampling.

        Args:
            input_value: The model input value.

        Returns:
            The acquisition criterion value.
        """
        return self._compute(input_value)

    def _compute_by_batch(self, q_input_values: NumberArray) -> NumberArray | float:
        """Compute the parallelized acquisition criterion value using a random process.

        Args:
            q_input_values: The batch of $q$ input values.

        Returns:
            The parallelized acquisition criterion value.
        """
        msg = "Parallelization is not defined for this acquisition criterion."
        raise NotImplementedError(msg)

    def _compute_by_batch_empirically(
        self, q_input_values: NumberArray
    ) -> NumberArray | float:
        """Compute the parallelized acquisition criterion using resampling.

        Args:
            q_input_values: The batch of $q$ input values.

        Returns:
            The parallelized acquisition criterion value.
        """
        msg = (
            "Parallelization with batch_size > 1 is not yet implemented "
            "for regressors that are not based on a random process."
        )
        raise NotImplementedError(msg)

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

    def _reshape_input_values(self, input_values: NumberArray) -> NumberArray:
        r"""Reshape an array from `(q\times d,)` to `(q, d)`.

        Args:
            input_values: The input values shaped as `(q\times d,)`.

        Returns:
            The input values shaped as `(q, d)`.
        """
        return input_values.reshape(self._batch_size, -1)

    def update(self) -> None:
        """Update the acquisition criterion."""
        data = self._regressor_distribution.learning_set.output_dataset.to_numpy()
        self.output_range = data.max() - data.min()

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

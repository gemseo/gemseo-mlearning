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
"""Acquisition criterion for which the optimum would improve the regression model.

An acquisition criterion (also called *infill criterion*)
is a function taking a model input value
and returning a value of interest to maximize (default option) or minimize
according to the meaning of the acquisition criterion.

Then, the input value optimizing this criterion can be used to enrich the dataset
used by a machine learning algorithm in its training stage.
This is the purpose of active learning.

This notion of acquisition criterion is implemented through the
[AcquisitionCriterion][gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion.BaseAcquisitionCriterion]
class which is built from a
[BaseRegressor][gemseo.mlearning.regression.algos.base_regressor.BaseRegressor]
and inherits from
[MDOFunction][gemseo.core.mdofunctions.mdo_function.MDOFunction].
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from numpy import ones

if TYPE_CHECKING:
    from gemseo.typing import NumberArray

    from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (  # noqa: E501
        BaseRegressorDistribution,
    )


class BaseAcquisitionCriterion(MDOFunction):
    """A base acquisition criterion."""

    algo_distribution: BaseRegressorDistribution
    """The distribution of a machine learning algorithm assessor."""

    output_range: float
    """The output range."""

    MAXIMIZE: ClassVar[bool] = True
    """Whether this criterion must be maximized."""

    def __init__(self, algo_distribution: BaseRegressorDistribution) -> None:
        """
        Args:
            algo_distribution: The distribution of a machine learning algorithm.
        """  # noqa: D205 D212 D415
        self.algo_distribution = algo_distribution
        dataset = self.algo_distribution.learning_set
        data = dataset.get_view(group_names=dataset.OUTPUT_GROUP).to_numpy()
        self.output_range = data.max() - data.min()
        try:
            jac = self._compute_jacobian(ones(algo_distribution.algo.input_dimension))
        except NotImplementedError:
            jac = None
        super().__init__(
            self._compute_output,
            self._compute_output.__name__,
            jac=jac,
        )

    @property
    def _scaling_factor(self) -> float:
        """The factor to scale values in the output space."""
        if self.output_range == 0:
            return 1.0

        return self.output_range

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

    def __truediv__(self, other: MDOFunction | float) -> MDOFunction:
        new_criterion = super().__truediv__(other)
        new_criterion.algo_distribution = self.algo_distribution
        new_criterion.output_range = self.output_range
        return new_criterion

    def __neg__(self) -> MDOFunction:
        new_criterion = super().__neg__()
        new_criterion.algo_distribution = self.algo_distribution
        new_criterion.output_range = self.output_range
        return new_criterion

    def __add__(self, other: MDOFunction | float) -> MDOFunction:
        new_criterion = super().__add__(other)
        new_criterion.algo_distribution = self.algo_distribution
        new_criterion.output_range = self.output_range
        return new_criterion

    def __sub__(self, other: MDOFunction | float) -> MDOFunction:
        new_criterion = super().__sub__(other)
        new_criterion.algo_distribution = self.algo_distribution
        new_criterion.output_range = self.output_range
        return new_criterion

    def __mul__(self, other: MDOFunction | float) -> MDOFunction:
        new_criterion = super().__mul__(other)
        new_criterion.algo_distribution = self.algo_distribution
        new_criterion.output_range = self.output_range
        return new_criterion

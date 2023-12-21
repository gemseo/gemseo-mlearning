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
This is the purpose of adaptive learning.

This notion of acquisition criterion is implemented through the
:class:`.MLDataAcquisitionCriterion` class which is built from a
:class:`.MLSupervisedAlgo` and inherits from :class:`.MDOFunction`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar

from gemseo.core.base_factory import BaseFactory
from gemseo.core.mdofunctions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo_mlearning.adaptive.distribution import MLRegressorDistribution

MLDataAcquisitionCriterionOptionType = float


class MLDataAcquisitionCriterion(MDOFunction):
    """Acquisition criterion."""

    algo_distribution: MLRegressorDistribution
    """The distribution of a machine learning algorithm assessor."""

    output_range: float
    """The output range."""

    MAXIMIZE: ClassVar[bool] = True

    def __init__(
        self,
        algo_distribution: MLRegressorDistribution,
        **options: MLDataAcquisitionCriterionOptionType,
    ) -> None:
        """
        Args:
            algo_distribution: The distribution of a machine learning algorithm.
            **options: The acquisition criterion options.
        """  # noqa: D205 D212 D415
        self.algo_distribution = algo_distribution
        dataset = self.algo_distribution.learning_set
        data = dataset.get_view(group_names=dataset.OUTPUT_GROUP).to_numpy()
        self.output_range = data.max() - data.min()
        func = self._get_func()
        super().__init__(func, func.__name__, jac=self._get_jac())

    @property
    def _scaling_factor(self) -> float:
        """The factor to scale values in the output space."""
        if self.output_range == 0:
            return 1.0

        return self.output_range

    @abstractmethod
    def _get_func(self) -> Callable:
        """Build the evaluation function.

        Returns:
            The evaluation function.
        """

    def _get_jac(self) -> Callable[[ndarray], ndarray] | None:
        """Return the Jacobian function if any.

        Returns:
            The Jacobian function if any.
        """

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


class MLDataAcquisitionCriterionFactory(BaseFactory):
    """A factory of :class:`.MLDataAcquisitionCriterion`."""

    _CLASS = MLDataAcquisitionCriterion
    _MODULE_NAMES = ("gemseo_mlearning.adaptive.criteria",)

    @property
    def available_criteria(self) -> list[str]:
        """The names of the available criteria."""
        return self.class_names

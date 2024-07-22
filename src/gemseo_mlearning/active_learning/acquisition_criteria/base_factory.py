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
"""A factory of acquisition criteria."""

from abc import abstractmethod
from typing import ClassVar

from gemseo.core.base_factory import BaseFactory

from gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion import (  # noqa: E501
    BaseAcquisitionCriterion,
)


class BaseAcquisitionCriterionFactory(BaseFactory):
    """A factory of acquisition criteria."""

    _CLASS = BaseAcquisitionCriterion
    _MODULE_NAMES = ("gemseo_mlearning.active_learning.acquisition_criteria",)

    @property
    @abstractmethod
    def _DEFAULT_CLASS_NAME(self) -> str:  # noqa: N802
        """The name of the default class."""

    def get_class(self, name: str = "") -> type[BaseAcquisitionCriterion]:  # noqa: D102
        return super().get_class(name or self._DEFAULT_CLASS_NAME)


class BaseAcquisitionCriterionFamilyFactory:
    """The factory of families of acquisition criteria."""

    ACQUISITION_CRITERION_FAMILY_FACTORY: ClassVar[BaseAcquisitionCriterionFactory]

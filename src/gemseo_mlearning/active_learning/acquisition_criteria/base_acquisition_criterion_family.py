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
"""Base class for families of acquisition criteria."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.core.base_factory import BaseFactory

if TYPE_CHECKING:
    from gemseo_mlearning.active_learning.acquisition_criteria.base_factory import (
        BaseAcquisitionCriterionFactory,
    )


class BaseAcquisitionCriterionFamily:
    """The base class for families of acquisition criteria."""

    ACQUISITION_CRITERION_FACTORY: ClassVar[type[BaseAcquisitionCriterionFactory]]


class AcquisitionCriterionFamilyFactory(BaseFactory):
    """The factory of families of acquisition criteria."""

    _CLASS = BaseAcquisitionCriterionFamily
    _MODULE_NAMES = ("gemseo_mlearning.active_learning.acquisition_criteria",)

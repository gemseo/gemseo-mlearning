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
"""A factory of acquisition criteria to approximate quantiles."""

from __future__ import annotations

from gemseo_mlearning.active_learning.acquisition_criteria.base_factory import (
    BaseAcquisitionCriterionFactory,
)
from gemseo_mlearning.active_learning.acquisition_criteria.quantile.base_quantile import (  # noqa: E501
    BaseQuantile,
)


class QuantileFactory(BaseAcquisitionCriterionFactory):
    """A factory of acquisition criteria to approximate level sets."""

    _CLASS = BaseQuantile
    _MODULE_NAMES = ("gemseo_mlearning.active_learning.acquisition_criteria.quantile",)
    _DEFAULT_CLASS_NAME = "U"

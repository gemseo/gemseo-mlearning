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
"""Family of acquisition criteria to estimate a maximum."""

from gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion_family import (  # noqa: E501
    BaseAcquisitionCriterionFamily,
)
from gemseo_mlearning.active_learning.acquisition_criteria.maximum.factory import (
    MaximumFactory,
)


class Maximum(BaseAcquisitionCriterionFamily):
    """The family of acquisition criteria to estimate a maximum."""

    ACQUISITION_CRITERION_FACTORY = MaximumFactory

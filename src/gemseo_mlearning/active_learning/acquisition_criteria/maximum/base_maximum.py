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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base class for acquisition criteria to estimate a maximum."""

from __future__ import annotations

from gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion import (  # noqa: E501
    BaseAcquisitionCriterion,
)


class BaseMaximum(BaseAcquisitionCriterion):
    """The base class for acquisition criteria to estimate a maximum."""

    def update(self) -> None:  # noqa: D102
        super().update()
        self._qoi = max(
            self._regressor_distribution.learning_set.output_dataset.to_numpy()
        )

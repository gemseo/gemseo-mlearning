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
"""Distance to the learning set."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import atleast_2d
from numpy import nonzero
from scipy.spatial.distance import cdist

from gemseo_mlearning.active_learning.acquisition_criteria.exploration.base_exploration import (  # noqa: E501
    BaseExploration,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class Distance(BaseExploration):
    """Distance to the learning set.

    This acquisition criterion computes the minimum distance between a new point and the
    point of the learning dataset, scaled by the minimum distance between two distinct
    learning points.
    """

    def _compute(self, input_value: NumberArray) -> NumberArray | float:  # noqa: D102
        train = self._regressor_distribution.learning_set.input_dataset.to_numpy()
        return (
            cdist(atleast_2d(input_value), train).min(axis=-1)
            / (dist_train := cdist(train, train))[nonzero(dist_train)].min()
            * 2.0
        )

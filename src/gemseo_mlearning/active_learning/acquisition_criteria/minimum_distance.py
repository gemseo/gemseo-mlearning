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
"""Criterion to be maximized for adding points away from the learning dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import nonzero
from scipy.spatial.distance import cdist

from gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion import (  # noqa: E501
    BaseAcquisitionCriterion,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class MinimumDistance(BaseAcquisitionCriterion):
    """Criterion to be maximized for adding points away from the learning dataset.

    This infill criterion computes the minimum distance between a new point and the
    point of the learning dataset, scaled by the maximum distance between two learning
    points.
    """

    def _compute_output(self, input_data: NumberArray) -> NumberArray:
        train = self.algo_distribution.learning_set
        train = train.get_view(group_names=train.INPUT_GROUP).to_numpy()
        distance = cdist(input_data.reshape((1, -1)), train).min()
        dist_train = cdist(train, train)
        d_max = dist_train[nonzero(dist_train)].min() / 2.0
        distance /= d_max
        return distance

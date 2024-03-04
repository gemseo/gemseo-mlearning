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
from __future__ import annotations

import pytest
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.linreg import LinearRegressor
from gemseo.utils.testing.helpers import concretize_classes
from numpy import array

from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (
    BaseRegressorDistribution,
)


@pytest.fixture(scope="module")
def algo_distribution() -> BaseRegressorDistribution:
    """A mock distribution of a regression model.

    This distribution uses mocks for the methods compute_variance and compute_mean.
    """
    dataset = IODataset()
    dataset.add_variable(
        "x", array([0.0, 0.5, 1.0])[:, None], group_name=dataset.INPUT_GROUP
    )
    dataset.add_variable(
        "y",
        array([1.0, 0.0, 1.0])[:, None],
        group_name=dataset.OUTPUT_GROUP,
    )
    with concretize_classes(BaseRegressorDistribution):
        distribution = BaseRegressorDistribution(LinearRegressor(dataset))

    distribution.learn()
    distribution.compute_variance = lambda input_data: 2 * input_data
    distribution.compute_mean = lambda input_data: 3 * input_data

    def expected_improvement(input_data, minimum, maximize=True):
        return 4 * input_data

    distribution.compute_expected_improvement = expected_improvement
    return distribution

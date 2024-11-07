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
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gemseo import sample_disciplines
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.algos.gpr import GaussianProcessRegressor

from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo

if TYPE_CHECKING:
    from gemseo.datasets.io_dataset import IODataset


@pytest.fixture(scope="module")
def discipline() -> AnalyticDiscipline:
    """The original discipline."""
    return AnalyticDiscipline({"z": "(1-x)**2+100*(y-x**2)**2"}, name="Rosenbrock")


@pytest.fixture(scope="module")
def input_space() -> DesignSpace:
    """The input space of the surrogate model."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-2, upper_bound=2, value=1.0)
    design_space.add_variable("y", lower_bound=-2, upper_bound=2, value=1.0)
    return design_space


@pytest.fixture(scope="module")
def learning_dataset(discipline, input_space) -> IODataset:
    """The learning dataset."""
    return sample_disciplines(
        [discipline], input_space, "z", algo_name="OT_OPT_LHS", n_samples=30
    )


@pytest.fixture(scope="module")
def active_learning_algo(
    input_space, learning_dataset, discipline
) -> ActiveLearningAlgo:
    """The active learning algorithm."""
    regressor = GaussianProcessRegressor(learning_dataset)
    algo = ActiveLearningAlgo("Minimum", input_space, regressor)
    algo.acquire_new_points(discipline)
    return algo

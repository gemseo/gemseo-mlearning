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

import re
from typing import TYPE_CHECKING

import pytest
from gemseo import sample_disciplines
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.algos.gpr import GaussianProcessRegressor
from gemseo.utils.platform import PLATFORM_IS_WINDOWS
from gemseo.utils.testing.helpers import image_comparison
from numpy import array

from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo
from gemseo_mlearning.active_learning.visualization.acquisition_view import (
    AcquisitionView,
)

if TYPE_CHECKING:
    from gemseo.datasets.io_dataset import IODataset

TOL = 0.0 if PLATFORM_IS_WINDOWS else 0.9


@pytest.fixture(scope="module")
def input_space() -> DesignSpace:
    """An input space."""
    input_space = DesignSpace()
    input_space.add_variable("x", l_b=-2, u_b=2, value=1.0)
    input_space.add_variable("y", l_b=-2, u_b=2, value=1.0)
    return input_space


@pytest.fixture(scope="module")
def discipline() -> AnalyticDiscipline:
    """The original discipline."""
    return AnalyticDiscipline({"z": "(1-x)**2+100*(y-x**2)**2"}, name="Rosenbrock")


@pytest.fixture(scope="module")
def learning_dataset(discipline, input_space) -> IODataset:
    """The learning dataset."""
    return sample_disciplines([discipline], input_space, "z", 30, "OT_OPT_LHS")


@pytest.fixture(scope="module")
def viewer(discipline, learning_dataset, input_space) -> AcquisitionView:
    """The active learning algorithm."""
    regressor = GaussianProcessRegressor(learning_dataset)
    algo = ActiveLearningAlgo("Minimum", input_space, regressor)
    algo.acquire_new_points(discipline)
    return AcquisitionView(algo)


def test_raise_parallel(discipline, learning_dataset, input_space):
    """Check that an error is raised when plotting in batch mode."""

    regressor = GaussianProcessRegressor(learning_dataset)
    algo = ActiveLearningAlgo("Minimum", input_space, regressor, batch_size=3)

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "AcquisitionView does not support active learning"
            " algorithm using parallel acquisition."
        ),
    ):
        AcquisitionView(algo)


@image_comparison(["default"], tol=0.9)
def test_default(viewer):
    """Check AcquisitionView with default settings."""
    viewer.draw(show=False)


@image_comparison(["custom"], tol=0.9)
def test_custom(viewer, discipline):
    """Check AcquisitionView with custom settings."""
    viewer.draw(
        show=False,
        discipline=discipline,
        new_point=array([0.0, 0.0]),
        filled=False,
        n_test=5,
    )


@image_comparison(["new_point"], tol=0.9)
def test_new_point(viewer):
    """Check AcquisitionView with a new point."""
    viewer.draw(show=False, new_point=array([0.0, 0.0]))


@image_comparison(["filled_false"], tol=TOL)
def test_filled(viewer):
    """Check AcquisitionView without filled contours."""
    viewer.draw(show=False, filled=False)


@image_comparison(["n_test"])
def test_n_test(viewer):
    """Check AcquisitionView with a lower number of points."""
    viewer.draw(show=False, n_test=5)


@image_comparison(["discipline"], tol=0.9)
def test_discipline(viewer, discipline):
    """Check AcquisitionView with a discipline."""
    viewer.draw(show=False, discipline=discipline)

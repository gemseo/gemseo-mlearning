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

import pytest
from gemseo.mlearning.regression.algos.gpr import GaussianProcessRegressor
from gemseo.utils.testing.helpers import image_comparison
from matplotlib.figure import Figure
from numpy import array

from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo
from gemseo_mlearning.active_learning.visualization.acquisition_view import (
    AcquisitionView,
)


@pytest.fixture(scope="module")
def acquisition_view(active_learning_algo) -> AcquisitionView:
    """The acquisition view."""
    return AcquisitionView(active_learning_algo)


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


def test_return_type(acquisition_view):
    """Check the return type of AcquisitionView."""
    assert isinstance(acquisition_view.draw(show=False), Figure)


@image_comparison(["default"], tol=0.9)
def test_default(acquisition_view):
    """Check AcquisitionView with default settings."""
    acquisition_view.draw(show=False)


@image_comparison(["custom"], tol=0.9)
def test_custom(acquisition_view, discipline):
    """Check AcquisitionView with custom settings."""
    acquisition_view.draw(
        show=False,
        discipline=discipline,
        new_point=array([0.0, 0.0]),
        filled=False,
        n_test=5,
    )


@image_comparison(["new_point"], tol=0.9)
def test_new_point(acquisition_view):
    """Check AcquisitionView with a new point."""
    acquisition_view.draw(show=False, new_point=array([0.0, 0.0]))


@image_comparison(["filled_false"], tol=0.9)
def test_filled(acquisition_view):
    """Check AcquisitionView without filled contours."""
    acquisition_view.draw(show=False, filled=False)


@image_comparison(["n_test"], tol=0.9)
def test_n_test(acquisition_view):
    """Check AcquisitionView with a lower number of points."""
    acquisition_view.draw(show=False, n_test=5)


@image_comparison(["discipline"], tol=0.9)
def test_discipline(acquisition_view, discipline):
    """Check AcquisitionView with a discipline."""
    acquisition_view.draw(show=False, discipline=discipline)

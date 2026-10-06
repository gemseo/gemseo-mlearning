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
from gemseo.machine_learning.regression.model.gpr import GaussianProcessRegressor
from gemseo.post.dataset import Lines

# TODO(bump-gemseo): gemseo.utils.testing.helpers.image_comparison: use the snapshot_matplotlib fixture of syrupy-matplotlib instead; add syrupy-matplotlib to the test dependencies, set snapshot_matplotlib_tolerance = 0.01 in .pytest.ini, replace the @image_comparison([...]) decorator by a snapshot_matplotlib argument of the test, then run pytest --snapshot-update without -n (the baseline_images directories are no longer used)  # noqa: E501
from gemseo.util.testing.helper import image_comparison

from gemseo_mlearning.active_learning.active_learning_algo import ActiveLearningAlgo
from gemseo_mlearning.active_learning.visualization.qoi_history_view import (
    QOIHistoryView,
)


@pytest.fixture(scope="module")
def qoi_history_view(active_learning_algo) -> QOIHistoryView:
    """The QOI history view."""
    return QOIHistoryView(active_learning_algo)


def test_return_type(qoi_history_view):
    """Check the return type of QOIHistoryView."""
    assert isinstance(qoi_history_view.draw(show=False), Lines)


@image_comparison(["default"])
def test_default(qoi_history_view):
    """Check QOIHistoryView with default settings."""
    qoi_history_view.draw(show=False)


@image_comparison(["custom"])
def test_custom(qoi_history_view):
    """Check QOIHistoryView with custom settings."""
    qoi_history_view.draw(show=False, plot_abscissa_variable=True, label="foo")


def test_error_no_qoi(learning_dataset, input_space):
    """Check that an error is raised when there is no QOI."""
    regressor = GaussianProcessRegressor(learning_dataset)
    active_learning_algo = ActiveLearningAlgo("Exploration", input_space, regressor)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "There is no quantity of interest "
            "associated with the acquisition criterion Variance."
        ),
    ):
        QOIHistoryView(active_learning_algo)

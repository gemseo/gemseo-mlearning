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
"""Tests for the surrogate-based optimization algorithms."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import pytest
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.problems.analytical.rastrigin import Rastrigin
from pandas.testing import assert_frame_equal


def test_default_options():
    """Check the default options of the surrogate-based optimizer."""
    assert OptimizersFactory().execute(Rastrigin(), "SBO").f_opt < 0.5


@pytest.mark.parametrize("max_iter", [8, 10])
def test_inconsistent_max_iter(max_iter, regression_algorithm):
    """Check that max_iter must be strictly greater than doe_size."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"max_iter ({max_iter}) must be "
            f"strictly greater than the initial DOE size (10)."
        ),
    ):
        OptimizersFactory().execute(Rastrigin(), "SBO", max_iter=max_iter)

    # Except if the regression algorithm is already built.
    OptimizersFactory().execute(
        Rastrigin(), "SBO", max_iter=max_iter, regression_algorithm=regression_algorithm
    )


def test_save(regression_algorithm, tmp_wd):
    """Check that the  regression algorithm can be pickled."""
    file_path = Path("model.pkl")
    OptimizersFactory().execute(
        Rastrigin(),
        "SBO",
        max_iter=3,
        acquisition_algorithm="OT_MONTE_CARLO",
        acquisition_options={"n_samples": 10},
        regression_algorithm=regression_algorithm,
        regression_file_path=file_path,
    )
    with file_path.open("rb") as file:
        model = pickle.load(file)

    assert_frame_equal(model.learning_set, regression_algorithm.learning_set)

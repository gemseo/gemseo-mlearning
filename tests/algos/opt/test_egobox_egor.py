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
from egobox import Egor
from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.problems.optimization.rosenbrock import Rosenbrock
from numpy.linalg import norm
from numpy.testing import assert_equal
from scipy.optimize import rosen

from gemseo_mlearning.algos.opt.egobox.constraint_strategy import ConstraintStrategy
from gemseo_mlearning.algos.opt.egobox.egobox_egor import EGOboxEgor
from gemseo_mlearning.algos.opt.egobox.egor_settings import EGObox_Egor_Settings

if TYPE_CHECKING:
    from gemseo.typing import RealArray

MAX_ITER = 50
N_SAMPLES = 10


@pytest.fixture
def problem() -> Rosenbrock:
    """The optimization problem."""
    return Rosenbrock()


def rosenbrock(x: RealArray) -> RealArray:
    """The rosenbrock function."""
    x = x * 4 - 2
    return rosen(x.T)[:, None]


@pytest.fixture
def reference(problem) -> RealArray | float:
    """The reference x_opt, f_opt and x_doe using egor with its default settings."""
    ego = Egor([[0.0, 1.0], [0.0, 1.0]], n_doe=N_SAMPLES, seed=0)
    result = ego.minimize(rosenbrock, max_iters=MAX_ITER - N_SAMPLES - 1)
    return result.x_opt * 4 - 2, result.y_opt, result.x_doe[:N_SAMPLES]


def test_default_settings(problem, reference):
    """Check that the default settings of EGObox_EGO and egor are identical."""
    result = EGOboxEgor().execute(
        problem, settings_model=EGObox_Egor_Settings(max_iter=50)
    )
    assert_equal(result.x_opt, reference[0])
    assert_equal(result.f_opt, reference[1])


@pytest.mark.parametrize("use_doe_settings", [False, True])
def test_doe(problem, reference, use_doe_settings):
    """Check the use of an initial DOE."""
    doe = reference[2] * 4 - 2
    if use_doe_settings:
        doe = CustomDOE_Settings(samples=doe)
    result = EGOboxEgor().execute(
        problem, settings_model=EGObox_Egor_Settings(max_iter=50, doe=doe)
    )
    assert_equal(result.x_opt, reference[0])
    assert_equal(result.f_opt, reference[1])


@pytest.mark.parametrize("max_iter", [10, 11])
def test_max_iter_error(problem, max_iter):
    """Check the exception raised when max_iter and n_samples are not consistent."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The maximum number of iterations must be strictly greater "
            "than 1 + n_initial_samples (=11)."
        ),
    ):
        EGOboxEgor().execute(
            problem, settings_model=EGObox_Egor_Settings(max_iter=max_iter)
        )


@pytest.mark.parametrize("cstr_strategy", ConstraintStrategy)
def test_constraints(problem, cstr_strategy):
    """Check that the three constraint strategies find f_opt < 1."""
    problem.add_constraint(
        MDOFunction(norm, "UnitDisk"),
        value=1.0,
        constraint_type=problem.ConstraintType.INEQ,
    )
    EGOboxEgor().execute(
        problem,
        settings_model=EGObox_Egor_Settings(max_iter=50, cstr_strategy=cstr_strategy),
    )
    assert problem.optimum.objective < 1

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
"""SMT optimizaton EGO tests."""

from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.problems.optimization.rosenbrock import Rosenbrock
from numpy import array
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from scipy.optimize import rosen
from smt.applications import EGO
from smt.surrogate_models import KRG
from smt.utils.design_space import DesignSpace
from smt.utils.design_space import DesignSpace as SMTDesignSpace

from gemseo_mlearning.algos.opt.smt._parallel_evaluator import ParallelEvaluator
from gemseo_mlearning.algos.opt.smt.ego_settings import AcquisitionCriterion
from gemseo_mlearning.algos.opt.smt.ego_settings import ParallelStrategy
from gemseo_mlearning.algos.opt.smt.ego_settings import SMT_EGO_Settings
from gemseo_mlearning.algos.opt.smt.ego_settings import Surrogate
from gemseo_mlearning.algos.opt.smt.smt_ego import SMTEGO

if TYPE_CHECKING:
    from gemseo.typing import RealArray


def scaled_rosenbrock(x: RealArray) -> float:
    """Evaluate the Rosenbrock function with normalized input data.

    Args:
        x: The input value.

    Returns:
        The output value of the Rosenbrock function.
    """
    return rosen(x * 4 - 2)


@pytest.mark.parametrize("criterion", AcquisitionCriterion)
@pytest.mark.parametrize("surrogate", Surrogate)
def test_criteria(criterion, surrogate):  # noqa: N803
    """Test EGO with different criteria."""
    if surrogate == Surrogate.MGP and criterion == AcquisitionCriterion.SBO:
        return

    optimization_problem = Rosenbrock()
    SMTEGO().execute(
        optimization_problem,
        n_start=2,
        n_doe=5,
        max_iter=7,
        n_max_optim=5,
        criterion=criterion,
        surrogate=surrogate,
    )
    last_eval = optimization_problem.objective.last_eval

    # Reference results using SMT directly.
    design_space = DesignSpace(array([[0.0, 1.0], [0.0, 1.0]]))
    surrogate = SMTEGO._SMTEGO__NAMES_TO_CLASSES[surrogate](design_space=design_space)
    ego = EGO(
        surrogate=surrogate,
        n_iter=1,
        n_doe=5,
        criterion=criterion,
        n_start=2,
        random_state=1,
        n_max_optim=2,
        evaluator=ParallelEvaluator(1),
    )
    objective = MDOFunction(scaled_rosenbrock, "scaled_rosenbrock")
    ego.optimize(fun=objective.evaluate)

    assert_allclose(objective.last_eval, last_eval, atol=1e-2)


@pytest.mark.parametrize(
    ("n_parallel", "x_opt", "f_opt"),
    [
        (1, array([-0.282595, 0.676449]), array([37.23683274])),
        (2, array([-0.736361, 1.068123]), array([30.670789])),
    ],
)
@pytest.mark.parametrize("normalize_design_space", [False, True])
def test_batch(n_parallel, x_opt, normalize_design_space, f_opt):  # noqa: N803
    """Test EGO w/wo batch acquisition and w/wo design space normalization."""
    optimization_problem = Rosenbrock()
    optimization_result = SMTEGO().execute(
        optimization_problem,
        settings_model=SMT_EGO_Settings(
            n_start=2,
            n_doe=5,
            max_iter=7,
            n_max_optim=5,
            n_parallel=n_parallel,
            normalize_design_space=normalize_design_space,
        ),
    )
    assert_allclose(optimization_result.x_opt, x_opt, atol=1e-5)
    assert_allclose(optimization_result.f_opt, f_opt, atol=1e-3)


@pytest.mark.parametrize("criterion", AcquisitionCriterion)
@pytest.mark.parametrize("qEI", ParallelStrategy)
def test_criteria_strategies(criterion, qEI):  # noqa: N803
    """Test EGO with different criteria and strategies.

    We mock the EGO class of SMT to check that the options are correctly passed.
    """
    optimization_problem = Rosenbrock()
    with (
        mock.patch.object(EGO, "__init__", return_value=None) as ego,
        contextlib.suppress(AttributeError),
    ):
        SMTEGO().execute(
            optimization_problem,
            n_start=2,
            n_doe=5,
            max_iter=7,
            n_max_optim=5,
            criterion=criterion,
            n_parallel=2,
            qEI=qEI,
            random_state=123,
        )

    assert ego.call_args.kwargs["n_start"] == 2
    assert ego.call_args.kwargs["n_doe"] == 5
    assert ego.call_args.kwargs["n_iter"] == 1
    assert ego.call_args.kwargs["n_max_optim"] == 5
    assert ego.call_args.kwargs["criterion"] == criterion
    assert ego.call_args.kwargs["n_parallel"] == 2
    assert ego.call_args.kwargs["qEI"] == qEI
    assert ego.call_args.kwargs["random_state"] == 123


@pytest.mark.parametrize(
    ("max_iter", "expected"),
    [
        (10, "max_iter must be strictly greater than n_doe+1 = 11; got 10."),
        (
            11,
            "max_iter must be strictly greater than n_doe+1 = 11; got 11.",
        ),
    ],
)
def test_max_iter(max_iter, expected):
    """Check that a ValueError is raised when max_iter is too small."""
    with pytest.raises(ValueError, match=re.escape(expected)):
        SMTEGO().execute(Rosenbrock(), max_iter=max_iter)


def test_from_existing_surrogate():
    """Test EGO from an existing surrogate."""
    result = SMTEGO().execute(
        Rosenbrock(),
        n_start=2,
        n_doe=5,
        max_iter=7,
        n_max_optim=5,
        surrogate="KRG",
    )
    x_opt = result.x_opt

    result = SMTEGO().execute(
        Rosenbrock(),
        n_start=2,
        n_doe=5,
        max_iter=7,
        n_max_optim=5,
        surrogate=KRG(
            # The design space is not [-2,2] but the unit hypercube,
            # because the optimization algorithm normalizes the design values.
            design_space=SMTDesignSpace(array([[0.0, 1.0], [0.0, 1.0]])),
            print_global=False,
        ),
    )

    assert_equal(result.x_opt, x_opt)

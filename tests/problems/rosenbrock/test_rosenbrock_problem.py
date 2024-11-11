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

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo_mlearning.problems.rosenbrock.rosenbrock_function import RosenbrockFunction
from gemseo_mlearning.problems.rosenbrock.rosenbrock_problem import RosenbrockProblem
from gemseo_mlearning.problems.rosenbrock.rosenbrock_space import RosenbrockSpace


@pytest.mark.parametrize("kwargs", [{}, {"use_uncertain_space": False}])
def test_rosenbrock_problem(kwargs) -> None:
    """Check the Rosenbrock problem."""
    problem = RosenbrockProblem(**kwargs)
    assert isinstance(problem.objective, RosenbrockFunction)

    input_space = problem.design_space
    if kwargs:
        assert input_space.dimension == 2
        assert input_space.variable_names == ["x1", "x2"]
        assert_equal(input_space.get_lower_bounds(), array([-2.0, -2.0]))
        assert_equal(input_space.get_upper_bounds(), array([2.0, 2.0]))
    else:
        assert isinstance(input_space, RosenbrockSpace)

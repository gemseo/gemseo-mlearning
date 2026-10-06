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

from numpy import array
from numpy.testing import assert_equal

from gemseo_mlearning.problems.rosenbrock.rosenbrock_function import RosenbrockFunction
from gemseo_mlearning.problems.rosenbrock.rosenbrock_problem import RosenbrockProblem


def test_rosenbrock_problem() -> None:
    """Check the Rosenbrock problem."""
    problem = RosenbrockProblem()
    assert isinstance(problem.objective, RosenbrockFunction)

    design_space = problem.design_space
    assert design_space.dimension == 2
    assert list(design_space.variables) == ["x1", "x2"]
    assert_equal(design_space.get_lower_bounds(), array([-2.0, -2.0]))
    assert_equal(design_space.get_upper_bounds(), array([2.0, 2.0]))
    assert_equal(design_space.get_current_value(), array([0.0, 0.0]))

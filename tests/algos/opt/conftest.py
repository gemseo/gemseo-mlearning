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
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.mlearning.regression.algos.ot_gpr import OTGaussianProcessRegressor
from gemseo.problems.optimization.rastrigin import Rastrigin


@pytest.fixture()
def regression_algorithm() -> OTGaussianProcessRegressor:
    """A regression algorithm for the Rastrigin problem."""
    problem = Rastrigin()
    DOELibraryFactory().execute(problem, "OT_SOBOL", n_samples=5)
    dataset = problem.to_dataset(opt_naming=False)
    dataset = dataset.map(lambda x: x.real)
    algo = OTGaussianProcessRegressor(
        dataset, transformer=OTGaussianProcessRegressor.DEFAULT_TRANSFORMER
    )
    algo.learn()
    return algo

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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re

import pytest
from gemseo.dataset import IODataset
from gemseo.machine_learning.regression.model.linreg import LinearRegressor
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings
from gemseo.util.testing.helper import assert_exception
from numpy import array
from numpy import corrcoef
from numpy.testing import assert_almost_equal
from openturns import CorrelationMatrix
from openturns import NormalCopula

from gemseo_mlearning.active_learning.acquisition_criteria.quantile.ef import EF
from gemseo_mlearning.active_learning.acquisition_criteria.quantile.ei import EI
from gemseo_mlearning.active_learning.acquisition_criteria.quantile.u import U
from gemseo_mlearning.active_learning.distributions import RegressorDistribution


@pytest.fixture(scope="module")
def uncertain_space() -> RandomSpace:
    """An uncertain space."""
    parameter_space = RandomSpace()
    parameter_space.add_variable("a", OTNormalDistribution_Settings())
    parameter_space.add_variable("x", OTNormalDistribution_Settings())
    return parameter_space


@pytest.mark.parametrize(
    ("cls", "input_value", "expected"),
    [
        (U, array([0.25]), array([8.5598195e-14])),
        (EF, array([0.25]), array([0.5746878])),
        (EI, array([0.25]), array([0.6843251])),
        (U, array([[0.25]] * 2), array([[8.5598195e-14]] * 2)),
        (EF, array([[0.25]] * 2), array([[0.5746878]] * 2)),
        (EI, array([[0.25]] * 2), array([[0.6843251]] * 2)),
    ],
)
def test_quantile_kriging(
    kriging_distribution, uncertain_space, cls, input_value, expected
):
    """Check the criteria deriving from BaseQuantile with a Kriging distribution."""
    criterion = cls(kriging_distribution, 0.8, uncertain_space)
    assert_almost_equal(criterion.func(input_value), expected)


@pytest.mark.parametrize(
    ("cls", "input_value", "expected"),
    [
        (U, array([0.25]), array([0.11785113])),
        (EF, array([0.25]), array([0.25])),
        (EI, array([0.25]), array([0.64])),
        (U, array([[0.25]] * 2), array([[0.11785113]] * 2)),
        (EF, array([[0.25]] * 2), array([[0.25]] * 2)),
        (EI, array([[0.25]] * 2), array([[0.64]] * 2)),
    ],
)
def test_quantile_regressor(
    algo_distribution, cls, input_value, expected, uncertain_space
):
    """Check the criteria deriving from BaseQuantile with non GP regressor."""
    criterion = cls(
        regressor_distribution=algo_distribution,
        level=0.8,
        uncertain_space=uncertain_space,
    )
    assert_almost_equal(criterion.func(input_value), expected, decimal=2)


@pytest.mark.parametrize(
    ("cls", "input_value", "expected"),
    [
        (U, array([[0.25]] * 2), array([1.0013774e-08])),
        (EF, array([[0.25]] * 2), 0.9428065),
        (EI, array([[0.25]] * 2), array([0.8888889])),
    ],
)
def test_quantile_parallel(
    kriging_distribution, cls, input_value, expected, uncertain_space
):
    """Check the parallelized criteria deriving from BaseQuantile."""
    criterion = cls(
        regressor_distribution=kriging_distribution,
        level=0.8,
        uncertain_space=uncertain_space,
        batch_size=2,
    )
    assert_almost_equal(criterion.func(input_value), expected)


def test_quantile_error(algo_distribution):
    """Check the exception raised by a BaseQuantile criterion."""
    uncertain_space = RandomSpace()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The probability distributions of the input variables x are missing."
        ),
    ):
        EF(algo_distribution, 0.8, uncertain_space)


@pytest.mark.parametrize(
    "cls",
    [
        U,
        EI,
        EF,
    ],
)
def test_bad_parallel_regressor(algo_distribution, cls, uncertain_space):
    """Check that parallelized criteria with non GP regressor lead to failure."""
    criterion = cls(
        regressor_distribution=algo_distribution,
        level=0.8,
        uncertain_space=uncertain_space,
        batch_size=2,
    )

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "Parallelization with batch_size > 1 is not yet implemented "
            "for regressors that are not based on a random process."
        ),
    ):
        criterion.func(array([0.123]))


def _create_normal_copula(correlation: float) -> NormalCopula:
    """Create a bivariate normal copula.

    Args:
        correlation: The correlation between the two variables.

    Returns:
        The normal copula.
    """
    matrix = CorrelationMatrix(2)
    matrix[0, 1] = correlation
    return NormalCopula(matrix)


@pytest.fixture(scope="module")
def two_input_distribution() -> RegressorDistribution:
    """A distribution of a regressor with the inputs x1 and x2."""
    dataset = IODataset()
    dataset.add_input_variable("x1", array([[0.0], [0.5], [1.0]]))
    dataset.add_input_variable("x2", array([[1.0], [0.0], [0.5]]))
    dataset.add_output_variable("y", array([[1.0], [0.0], [1.0]]))
    distribution = RegressorDistribution(LinearRegressor(dataset))
    distribution.learn()
    return distribution


def test_quantile_copula(two_input_distribution):
    """Check that a BaseQuantile criterion keeps the copulas of the inputs."""
    uncertain_space = RandomSpace()
    for name in ["x2", "a", "x1"]:
        uncertain_space.add_variable(name, OTNormalDistribution_Settings())

    uncertain_space.add_copula(["x2", "x1"], _create_normal_copula(0.9))
    criterion = EF(two_input_distribution, 0.8, uncertain_space, n_samples=10_000)
    input_data = criterion._BaseQuantile__input_data
    assert input_data.shape == (10_000, 2)
    assert corrcoef(input_data.T)[0, 1] > 0.85


def test_quantile_copula_error(two_input_distribution, snapshot):
    """Check that a copula linking an input with a non-input raises an error."""
    uncertain_space = RandomSpace()
    for name in ["x1", "x2", "a"]:
        uncertain_space.add_variable(name, OTNormalDistribution_Settings())

    uncertain_space.add_copula(["x1", "a"], _create_normal_copula(0.9))
    with assert_exception(ValueError, snapshot):
        EF(two_input_distribution, 0.8, uncertain_space)

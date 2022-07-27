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

import pytest
from gemseo.mlearning.regression.gpr import GaussianProcessRegressor
from gemseo_mlearning.adaptive.distributions.kriging_distribution import (
    KrigingDistribution,
)
from numpy import array


@pytest.fixture(scope="module")
def algo(dataset) -> GaussianProcessRegressor:
    """A Kriging regression model used in different tests."""
    return GaussianProcessRegressor(dataset)


@pytest.fixture(scope="module")
def distribution(algo: GaussianProcessRegressor) -> KrigingDistribution:
    """The distribution of the Kriging regression model."""
    distribution = KrigingDistribution(algo)
    distribution.learn()
    return distribution


def test_init(algo):
    """Check the initialization of the distribution."""
    assert KrigingDistribution(algo).algo == algo


@pytest.mark.parametrize(
    "point",
    [
        array([0.0]),
        array([0.5]),
        array([1.0]),
        array([[0.0], [0.5]]),
        {"x": array([0.0])},
        {"x": array([[0.0], [0.5]])},
    ],
)
def test_compute_mean(distribution, point):
    """Check the computation of the mean."""
    result = distribution.compute_mean(point)
    if isinstance(point, dict):
        result = result["y"]
        point = point["x"]

    assert result.shape == point.shape
    assert result[0] == pytest.approx(
        1 if point[0] in [0.0, 1.0] else 0.0,
        0.1,
    )


@pytest.mark.parametrize(
    "point",
    [
        array([0.0]),
        array([0.5]),
        array([1.0]),
        array([[0.0], [0.5]]),
        {"x": array([0.0])},
        {"x": array([[0.0], [0.5]])},
    ],
)
def test_compute_variance(distribution, point):
    """Check the computation of the variance."""
    result = distribution.compute_variance(point)
    if isinstance(point, dict):
        point = point["x"]
        result = result["y"]
    assert result.shape == point.shape
    assert result[0] == pytest.approx(0, abs=1e-3)


@pytest.mark.parametrize("maximize", [True, False])
@pytest.mark.parametrize(
    "point",
    [
        (array([0.0])),
        (array([1.0])),
        (array([[0.0], [1.0]])),
        ({"x": array([0.0])}),
        ({"x": array([[0.0], [0.5]])}),
    ],
)
def test_compute_expected_improvement(distribution, point, maximize):
    """Check the computation of the expected improvement."""
    result = distribution.compute_expected_improvement(point, 0.0, maximize)
    if isinstance(point, dict):
        point = point["x"]
        result = result["y"]

    assert result.shape == point.shape
    assert result[0] == pytest.approx(1 if maximize else 0, 0.1)


@pytest.mark.parametrize(
    "point",
    [
        array([0.0]),
        array([0.5]),
        array([1.0]),
        array([[0.0], [0.5]]),
        {"x": array([0.0])},
        {"x": array([[0.0], [0.5]])},
    ],
)
def test_confidence_interval(distribution, point):
    """Check the computation of a confidence interval."""
    lower, upper = distribution.compute_confidence_interval(point)
    if isinstance(point, dict):
        point = point["x"]
        lower = lower["y"]
        upper = upper["y"]
    assert (lower < upper).all()
    assert lower.shape == upper.shape == point.shape

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
from gemseo.mlearning.regression.linreg import LinearRegressor
from gemseo_mlearning.adaptive.distributions.regressor_distribution import (
    RegressorDistribution,
)
from numpy import array
from numpy import exp
from numpy import quantile


@pytest.fixture(scope="module")
def algo(dataset) -> LinearRegressor:
    """A linear regression model used in different tests."""
    model = LinearRegressor(dataset)
    model.learn()
    return model


def __weight_func(value, indices):
    terms = [
        1 - exp(-((value - 0.0) ** 2) / 0.5**2),
        1 - exp(-((value - 0.5) ** 2) / 0.5**2),
        1 - exp(-((value - 1.0) ** 2) / 0.5**2),
    ]
    result = 1
    for index in indices:
        result *= terms[index]
    return result


@pytest.fixture(scope="module")
def distribution(algo) -> RegressorDistribution:
    """The distribution of the linear regression model."""
    distribution = RegressorDistribution(algo, bootstrap=False, loo=True)
    distribution.learn()
    return distribution


@pytest.mark.parametrize(
    "bootstrap,loo,size",
    [
        (True, False, None),
        (True, False, 3),
        (False, False, None),
        (False, False, 3),
        (False, True, None),
    ],
)
def test_init(algo, bootstrap, loo, size):
    """Check the initialization of the distribution."""
    distribution = RegressorDistribution(algo, bootstrap, loo, size)
    if bootstrap:
        assert distribution.method == distribution.BOOTSTRAP
        assert distribution.size == size or distribution.N_BOOTSTRAP
    elif loo:
        assert distribution.method == distribution.LOO
        assert distribution.size == len(distribution.learning_set)
    else:
        assert distribution.method == distribution.CROSS_VALIDATION
        assert distribution.size == size or distribution.N_FOLDS

    assert len(distribution.algos) == distribution.size
    for algo in distribution.algos:
        assert isinstance(algo, LinearRegressor)


def test_learn(distribution):
    """Check the results of the learning stage for the original model.

    Original model: f(x) = 2/3
    Sub-models: f0(x) = -1 + 2x, f1(x) = 1, f2(x) = 1 - 2x
    """
    assert 2.0 / 3 == pytest.approx(distribution.algo.intercept[0], 0.1)
    assert 0.0 == pytest.approx(distribution.algo.coefficients[0], 0.1)


@pytest.mark.parametrize(
    "model,intercept,coefficient", [(0, -1, 2.0), (1, 1.0, 0.0), (2, 1.0, -2.0)]
)
def test_learn_submodels(distribution, model, intercept, coefficient):
    """Check the results of the learning stage for the sub-models.

    Sub-models: f0(x) = -1 + 2x, f1(x) = 1, f2(x) = 1 - 2x
    """
    assert intercept == pytest.approx(distribution.algos[model].intercept[0], 0.1)
    assert coefficient == pytest.approx(distribution.algos[model].coefficients[0], 0.1)


@pytest.mark.parametrize("point", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("model", [0, 1, 2])
def test_weight_functions(distribution, point, model):
    """Check the weight functions."""
    func = distribution.weights[model]
    assert __weight_func(point, [model]) == func(array([point]))
    assert __weight_func(point, [model]) == func({"x": array([point])})


@pytest.mark.parametrize("model", [0, 1, 2])
def test_weight_function_with_several_points(distribution, model):
    """Check the weight functions with several points at once."""
    points = array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected = array([__weight_func(point, [model]) for point in points])
    returned = distribution.weights[model](points.reshape((-1, 1)))
    assert returned.shape == (5,)
    assert sum(abs(expected - returned)) == 0


@pytest.mark.parametrize("point", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_evaluate_weights(distribution, point):
    """Check the evaluate function."""
    assert distribution._evaluate_weights(array([point])).shape == (3,)


def test_evaluate_weights_with_several_points(distribution):
    """Check the evaluate function with several points at once."""
    weights = distribution._evaluate_weights(array([[0.0, 0.25, 0.5, 0.75, 1.0]]).T)
    assert weights.shape == (3, 5)
    for value in weights.sum(0):
        assert pytest.approx(value, 0.1) == 1


@pytest.mark.parametrize(
    "point,outputs",
    [
        (array([0.0]), [-1, 1, 1]),
        ({"x": array([0.0])}, [-1, 1, 1]),
        (array([[0.0], [1.0]]), [[-1, 1, 1], [1, 1, -1]]),
        ({"x": array([[0.0], [1.0]])}, [[-1, 1, 1], [1, 1, -1]]),
    ],
)
def test_predict_members(distribution, point, outputs):
    """Check the predictions of the members."""
    predictions = distribution.predict_members(point)
    if isinstance(point, dict):
        predictions = predictions["y"]
        point = point["x"]

    n_points = point.shape[0]
    assert predictions.shape == (3, 1) if n_points == 1 else (3, n_points, 1)
    if n_points == 1:
        for model in [0, 1, 2]:
            assert predictions[model, 0] == pytest.approx(outputs[model], 0.1)
    else:
        for model in [0, 1, 2]:
            for point in range(n_points):
                expected = pytest.approx(outputs[point][model], 0.1)
                assert predictions[model, point, 0] == expected


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
        point = point["x"]
        result = result["y"]

    assert result.shape == point.shape
    assert result[0] == pytest.approx(0 if point[0] == 0.5 else 1, 0.1)


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
    assert result[0] == pytest.approx(0, 0.1)


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
    "point,outputs",
    [
        (array([0.0]), [array([-1, 1, 1])]),
        (array([1.0]), [array([-1, 1, 1])]),
        (
            array([[0.0], [0.5], [1.0]]),
            [array([-1, 1, 1]), array([0, 0, 1]), array([-1, 1, 1])],
        ),
        ({"x": array([0.0])}, [array([-1, 1, 1])]),
        (
            {"x": array([[0.0], [0.5], [1.0]])},
            [array([-1, 1, 1]), array([0, 0, 1]), array([-1, 1, 1])],
        ),
    ],
)
def test_compute_confidence_interval(distribution, point, outputs):
    """Check the computation of a confidence interval."""
    lower, upper = distribution.compute_confidence_interval(point)
    if isinstance(point, dict):
        point = point["x"]
        lower = lower["y"]
        upper = upper["y"]

    assert lower.shape == upper.shape == point.shape

    for index, output in enumerate(outputs):
        assert pytest.approx(lower[index], 0.1) == quantile(output, 0.025)
        assert pytest.approx(upper[index], 0.1) == quantile(output, 0.975)

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
"""Test the interface to the OpenTURNS' Kriging."""

from __future__ import annotations

from unittest import mock
from unittest.mock import Mock

import openturns
import pytest
from gemseo import execute_algo
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.datasets.io_dataset import IODataset
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from numpy import array
from numpy import hstack
from numpy import ndarray
from numpy import zeros
from numpy.testing import assert_allclose
from openturns import CovarianceMatrix
from openturns import GeneralizedExponential
from openturns import KrigingAlgorithm
from openturns import KrigingResult
from openturns import MaternModel
from openturns import NLopt
from packaging import version
from scipy.optimize import rosen

from gemseo_mlearning.regression.ot_gpr import OTGaussianProcessRegressor

OTGaussianProcessRegressor.HMATRIX_ASSEMBLY_EPSILON = 1e-10
OTGaussianProcessRegressor.HMATRIX_RECOMPRESSION_EPSILON = 1e-10
# The EPSILONs are reduced to make the HMAT-based Kriging interpolating.

OTGaussianProcessRegressor.MAX_SIZE_FOR_LAPACK = 9
# The maximum learning sample size to use LAPACK is reduced to accelerate the tests.


def func(x: ndarray) -> float:
    """Return the sum of the components of a vector.

    Args:
        x: A vector.

    Returns:
        The sum of the components of the vector.
    """
    return sum(x)


@pytest.fixture(scope="module")
def problem() -> Rosenbrock:
    """The Rosenbrock problem with an observable summing the inputs."""
    rosenbrock = Rosenbrock()
    rosenbrock.add_observable(MDOFunction(func, "sum"))
    return rosenbrock


@pytest.fixture(scope="module")
def dataset(problem) -> IODataset:
    """A 9-length full-factorial sampling of the Rosenbrock problem."""
    execute_algo(problem, "fullfact", n_samples=9, algo_type="doe")
    return problem.to_dataset(opt_naming=False)


@pytest.fixture(scope="module")
def dataset_2(problem) -> IODataset:
    """A 9-length full-factorial sampling of the Rosenbrock problem."""
    execute_algo(problem, "fullfact", n_samples=9, algo_type="doe")
    data = problem.to_dataset(opt_naming=False)
    data.add_variable(
        "rosen2",
        hstack((
            data.get_view(variable_names="rosen").to_numpy(),
            -data.get_view(variable_names="rosen").to_numpy(),
        )),
        group_name=data.OUTPUT_GROUP,
    )
    return data


@pytest.fixture(scope="module")
def kriging(dataset) -> OTGaussianProcessRegressor:
    """A Kriging model trained on the Rosenbrock dataset."""
    model = OTGaussianProcessRegressor(dataset)
    model.learn()
    return model


def test_class_constants(kriging):
    """Check the class constants."""
    assert kriging.LIBRARY == "OpenTURNS"
    assert kriging.SHORT_ALGO_NAME == "GPR"


@pytest.mark.parametrize(
    ("n_samples", "use_hmat"),
    [
        (1, False),
        (OTGaussianProcessRegressor.MAX_SIZE_FOR_LAPACK, False),
        (OTGaussianProcessRegressor.MAX_SIZE_FOR_LAPACK + 1, True),
    ],
)
def test_kriging_use_hmat_default(n_samples, use_hmat):
    """Check the default library (LAPACK or HMAT) according to the sample size."""
    dataset = IODataset.from_array(
        zeros((n_samples, 2)),
        variable_names=["in", "out"],
        variable_names_to_group_names={
            "in": IODataset.INPUT_GROUP,
            "out": IODataset.OUTPUT_GROUP,
        },
    )
    assert OTGaussianProcessRegressor(dataset).use_hmat is use_hmat


@pytest.mark.parametrize("use_hmat", [True, False])
def test_kriging_use_hmat(dataset, use_hmat):
    """Check that the HMAT can be specified at initialization or after."""
    kriging = OTGaussianProcessRegressor(dataset, use_hmat=use_hmat)
    # Check at initialization
    assert kriging.use_hmat is use_hmat

    # Check after initialization
    kriging.use_hmat = not use_hmat
    assert kriging.use_hmat is not use_hmat


def test_kriging_predict_on_learning_set(dataset):
    """Check that the Kriging interpolates the learning set."""
    kriging = OTGaussianProcessRegressor(dataset)
    kriging.learn()
    for x in kriging.learning_set.get_view(
        group_names=IODataset.INPUT_GROUP
    ).to_numpy():
        prediction = kriging.predict({"x": x})
        assert_allclose(prediction["sum"], sum(x), atol=1e-3)
        assert_allclose(prediction["rosen"], rosen(x))


@pytest.mark.parametrize("x1", [-1, 1])
@pytest.mark.parametrize("x2", [-1, 1])
def test_kriging_predict(dataset, x1, x2):
    """Check that the Kriging is not yet good enough to extrapolate."""
    kriging = OTGaussianProcessRegressor(dataset)
    kriging.learn()
    x = array([x1, x2])
    prediction = kriging.predict({"x": x})
    assert prediction["sum"] != pytest.approx(sum(x))
    assert prediction["rosen"] != pytest.approx(rosen(x))


@pytest.mark.parametrize("transformer", [None, {"inputs": "MinMaxScaler"}])
def test_kriging_predict_std_on_learning_set(transformer, dataset):
    """Check that the standard deviation is correctly predicted for a learning point.

    The standard deviation should be equal to zero.
    """
    kriging = OTGaussianProcessRegressor(dataset, transformer=transformer)
    kriging.learn()
    for x in kriging.learning_set.get_view(
        group_names=IODataset.INPUT_GROUP
    ).to_numpy():
        assert_allclose(kriging.predict_std(x), 0, atol=1e-1)


@pytest.mark.parametrize("x1", [-1, 1])
@pytest.mark.parametrize("x2", [-1, 1])
@pytest.mark.parametrize("transformer", [None, {"inputs": "MinMaxScaler"}])
def test_kriging_predict_std(transformer, dataset, x1, x2):
    """Check that the standard deviation is correctly predicted for a validation point.

    The standard deviation should be the square root of the variance computed by the
    method KrigingResult.getConditionalCovariance of OpenTURNS.
    """
    kriging = OTGaussianProcessRegressor(dataset, transformer=transformer)
    original_method = KrigingResult.getConditionalCovariance
    v1 = 4.0 + x1 + x2
    v2 = 9.0 + x1 + x2
    KrigingResult.getConditionalCovariance = Mock(
        return_value=CovarianceMatrix(2, [v1, 0.5, 0.5, v2])
    )
    kriging.learn()
    assert_allclose(kriging.predict_std(array([x1, x2])), array([v1, v2]) ** 0.5)
    KrigingResult.getConditionalCovariance = original_method


def test_kriging_predict_jacobian(kriging):
    """Check the shape of the Jacobian."""
    jacobian = kriging.predict_jacobian(array([[0.0, 0.0], [-2.0, -2.0], [2.0, 2.0]]))
    assert jacobian.shape == (3, 2, 2)


@pytest.mark.parametrize("output_name", ["rosen", "rosen2"])
@pytest.mark.parametrize(
    "input_data",
    [
        array([1.0, 1.0]),
        array([[1.0, 1.0]]),
        {"x": array([1.0, 1.0])},
        {"x": array([[1.0, 1.0]])},
    ],
)
def test_kriging_std_output_dimension(dataset_2, output_name, input_data):
    """Check the shape of the array returned by predict_std()."""
    model = OTGaussianProcessRegressor(dataset_2, output_names=[output_name])
    model.learn()
    ndim = model.output_dimension
    if isinstance(input_data, dict):
        one_sample = input_data["x"].ndim == 1
    else:
        one_sample = input_data.ndim == 1
    shape = (ndim,) if one_sample else (1, ndim)

    assert model.predict_std(input_data).shape == shape


@pytest.mark.parametrize(
    ("trend_type", "shape"),
    [
        (OTGaussianProcessRegressor.TrendType.CONSTANT, (2, 1)),
        (OTGaussianProcessRegressor.TrendType.LINEAR, (2, 3)),
        (OTGaussianProcessRegressor.TrendType.QUADRATIC, (2, 6)),
    ],
)
def test_trend_type(dataset, trend_type, shape):
    """Check the trend type of the Gaussian process regressor."""
    model = OTGaussianProcessRegressor(dataset, trend_type=trend_type)
    model.learn()
    if version.parse(openturns.__version__) < version.parse("1.21"):
        assert array(model.algo.getTrendCoefficients()).shape == shape
    else:
        assert array(model.algo.getTrendCoefficients()).shape == (shape[0] * shape[1],)


def test_default_optimizer(dataset):
    """Check that the default optimizer is TNC."""
    model = OTGaussianProcessRegressor(dataset)
    with mock.patch.object(KrigingAlgorithm, "setOptimizationAlgorithm") as method:
        model.learn()

    assert method.call_args.args[0].__class__.__name__ == "TNC"


def test_custom_optimizer(dataset):
    """Check that the optimizer can be changed."""
    optimizer = NLopt("LN_NELDERMEAD")
    model = OTGaussianProcessRegressor(dataset, optimizer=optimizer)
    with mock.patch.object(KrigingAlgorithm, "setOptimizationAlgorithm") as method:
        model.learn()

    assert method.call_args.args[0] == optimizer


def test_default_covariance_model(dataset):
    """Check default covariance model is SquaredExponential."""
    model = OTGaussianProcessRegressor(dataset)
    model.learn()
    assert "SquaredExponential" in str(model.algo.getCovarianceModel())


@pytest.mark.parametrize(
    ("covariance_model", "use_generalized"),
    [
        (MaternModel, False),
        (MaternModel(2), False),
        ([MaternModel, GeneralizedExponential], True),
        ([MaternModel(2), GeneralizedExponential(2)], True),
    ],
)
def test_custom_covariance_model(dataset, covariance_model, use_generalized):
    """Check that the covariance model can be changed."""
    model = OTGaussianProcessRegressor(dataset, covariance_model=covariance_model)
    model.learn()
    covariance_model_str = str(model.algo.getCovarianceModel())
    assert "MaternModel" in covariance_model_str
    if use_generalized:
        assert "GeneralizedExponential" in covariance_model_str

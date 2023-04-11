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

import pytest
from gemseo import execute_algo
from gemseo.core.dataset import Dataset
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo_mlearning.regression.ot_gpr import OTGaussianProcessRegressor
from numpy import array
from numpy import hstack
from numpy import ndarray
from numpy import zeros
from numpy.testing import assert_allclose
from scipy.optimize import rosen

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
def dataset(problem) -> Dataset:
    """A 9-length full-factorial sampling of the Rosenbrock problem."""
    execute_algo(problem, "fullfact", n_samples=9, algo_type="doe")
    return problem.to_dataset(opt_naming=False)


@pytest.fixture(scope="module")
def dataset_2(problem) -> Dataset:
    """A 9-length full-factorial sampling of the Rosenbrock problem."""
    execute_algo(problem, "fullfact", n_samples=9, algo_type="doe")
    data = problem.to_dataset(opt_naming=False)
    data.add_variable(
        "rosen2", hstack((data["rosen"], -data["rosen"])), group=data.OUTPUT_GROUP
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
    "n_samples,use_hmat",
    [
        (1, False),
        (OTGaussianProcessRegressor.MAX_SIZE_FOR_LAPACK, False),
        (OTGaussianProcessRegressor.MAX_SIZE_FOR_LAPACK + 1, True),
    ],
)
def test_kriging_use_hmat_default(n_samples, use_hmat):
    """Check the default library (LAPACK or HMAT) according to the sample size."""
    dataset = Dataset()
    dataset.set_from_array(
        zeros((n_samples, 2)),
        variables=["in", "out"],
        groups={"in": Dataset.INPUT_GROUP, "out": Dataset.OUTPUT_GROUP},
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
    for x in kriging.learning_set.get_data_by_group(Dataset.INPUT_GROUP):
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
    """Check that the Kriging interpolates the learning set.

    The standard deviation should be equal to zero.
    """
    kriging = OTGaussianProcessRegressor(dataset, transformer=transformer)
    kriging.learn()
    for x in kriging.learning_set.get_data_by_group(Dataset.INPUT_GROUP):
        predicted_std = kriging.predict_std(x)
        assert predicted_std[0] == pytest.approx(0.0, abs=1e-1)
        assert predicted_std[1] == pytest.approx(0.0, abs=1e-1)


@pytest.mark.parametrize("x1", [-1, 1])
@pytest.mark.parametrize("x2", [-1, 1])
@pytest.mark.parametrize("transformer", [None, {"inputs": "MinMaxScaler"}])
def test_kriging_predict_std(transformer, dataset, x1, x2):
    """Check that the Kriging is not yet good enough to extrapolate.

    The standard deviation should be different from zero.
    """
    kriging = OTGaussianProcessRegressor(dataset, transformer=transformer)
    kriging.learn()
    assert kriging.predict_std(array([x1, x2]))[0] != pytest.approx(0.0)
    assert kriging.predict_std(array([x1, x2]))[1] != pytest.approx(0.0)


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
    if one_sample:
        shape = (ndim,)
    else:
        shape = (1, ndim)

    assert model.predict_std(input_data).shape == shape

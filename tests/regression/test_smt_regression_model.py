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
from gemseo.datasets.io_dataset import IODataset
from numpy import array
from numpy import linspace
from numpy import newaxis
from numpy.testing import assert_almost_equal
from smt.surrogate_models import RBF

from gemseo_mlearning.regression.smt_regression_model import SMTRegressionModel


def test_smt_regression_model():
    """Check SMTRegressionModel."""
    input_data = linspace(-1, 1, 10)[:, newaxis]
    output_data = input_data**2

    dataset = IODataset()
    dataset.add_input_group(input_data)
    dataset.add_output_group(output_data)

    model = SMTRegressionModel(dataset, "RBF", d0=2)
    model.learn()

    smt_model = RBF(d0=2)
    smt_model.set_training_values(input_data, output_data)
    smt_model.train()

    input_value = array([0.25])
    assert_almost_equal(
        model.predict(input_value), smt_model.predict_values(input_value).ravel()
    )

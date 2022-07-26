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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Matthias
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.core.dataset import Dataset
from gemseo.problems.dataset.rosenbrock import RosenbrockDataset
from numpy import hstack
from numpy import ndarray


@pytest.fixture(scope="module")
def dataset() -> RosenbrockDataset:
    """The Rosenbrock dataset."""
    return RosenbrockDataset(opt_naming=False)


@pytest.fixture(scope="module")
def dataset_2(dataset) -> RosenbrockDataset:
    """The Rosenbrock dataset with 2d-output."""
    data = Dataset()
    data.add_variable("x", dataset["x"], group=data.INPUT_GROUP)
    data.add_variable("rosen", dataset["rosen"], group=data.OUTPUT_GROUP)
    data.add_variable(
        "rosen2", hstack((dataset["rosen"], dataset["rosen"])), group=data.OUTPUT_GROUP
    )
    return data


@pytest.fixture(scope="module")
def input_data(dataset) -> ndarray:
    """The learning input data."""
    return dataset.get_data_by_group(dataset.INPUT_GROUP)


@pytest.fixture(scope="module")
def output_data(dataset) -> ndarray:
    """The learning output data."""
    return dataset.get_data_by_group(dataset.OUTPUT_GROUP)

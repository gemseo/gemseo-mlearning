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
from gemseo.core.dataset import Dataset
from numpy import array


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """A learning dataset with three samples."""
    dataset = Dataset()
    dataset.add_variable(
        "x", array([0.0, 0.5, 1.0])[:, None], group=dataset.INPUT_GROUP
    )
    dataset.add_variable(
        "y",
        array([1.0, 0.0, 1.0])[:, None],
        group=dataset.OUTPUT_GROUP,
        cache_as_input=False,
    )
    return dataset

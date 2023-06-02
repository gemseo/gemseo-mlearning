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
"""Check the module api.py."""
from __future__ import annotations

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo_mlearning.api import sample_discipline
from gemseo_mlearning.api import sample_disciplines
from numpy import array
from numpy.testing import assert_equal


@pytest.fixture(scope="module")
def discipline() -> AnalyticDiscipline:
    """A simple discipline to be sampled, with 1 input and 2 outputs."""
    return AnalyticDiscipline({"out1": "2*inpt", "out2": "3*inpt"})


@pytest.fixture(scope="module")
def disciplines() -> list[AnalyticDiscipline]:
    """Two simple disciplines to be sampled, with 1 input and 2 outputs."""
    return [
        AnalyticDiscipline({"out1": "2*inpt"}),
        AnalyticDiscipline({"out2": "3*inpt"}),
    ]


@pytest.fixture(scope="module")
def input_space() -> DesignSpace:
    """The input space on which to sample the discipline."""
    design_space = DesignSpace()
    design_space.add_variable("inpt", l_b=1.0, u_b=2.0)
    return design_space


@pytest.mark.parametrize(
    "outputs",
    ["out1", ["out1", "out2"]],
)
@pytest.mark.parametrize("formulation", ["DisciplinaryOpt", "MDF"])
@pytest.mark.parametrize("name", [None, "foo"])
def test_sample_disciplines(disciplines, input_space, outputs, formulation, name):
    """Check the sampling of two disciplines."""
    dataset = sample_disciplines(
        disciplines, formulation, input_space, outputs, "fullfact", 2, name=name
    )
    if name is None:
        name = "DOEScenario"
    assert dataset.name == name

    assert_equal(
        dataset.get_view(variable_names="inpt").to_numpy(), array([[1.0], [2.0]])
    )

    if isinstance(outputs, str):
        outputs = [outputs]

    if "out1" in outputs:
        assert_equal(
            dataset.get_view(variable_names="out1").to_numpy(), array([[2.0], [4.0]])
        )

    if "out2" in outputs:
        assert_equal(
            dataset.get_view(variable_names="out2").to_numpy(), array([[3.0], [6.0]])
        )


@pytest.mark.parametrize(
    "outputs",
    ["out1", ["out1", "out2"]],
)
@pytest.mark.parametrize("name", [None, "foo"])
def test_sample_discipline(discipline, input_space, outputs, name):
    """Check the sampling of a discipline."""
    dataset = sample_discipline(
        discipline, input_space, outputs, "fullfact", 2, name=name
    )
    assert dataset.name == name or discipline.name

    assert_equal(
        dataset.get_view(variable_names="inpt").to_numpy(), array([[1.0], [2.0]])
    )

    if isinstance(outputs, str):
        outputs = [outputs]

    if "out1" in outputs:
        assert_equal(
            dataset.get_view(variable_names="out1").to_numpy(), array([[2.0], [4.0]])
        )

    if "out2" in outputs:
        assert_equal(
            dataset.get_view(variable_names="out2").to_numpy(), array([[3.0], [6.0]])
        )

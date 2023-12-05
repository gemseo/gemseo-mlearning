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
"""Compatibility between different versions of openturns."""

from __future__ import annotations

import openturns
from openturns import AggregatedFunction
from openturns import Basis
from openturns import BasisFactory
from packaging import version

if version.parse(openturns.__version__) < version.parse("1.21"):  # pragma: no cover

    def create_trend_basis(  # noqa: D103
        basis_factory: type(BasisFactory), input_dimension: int, output_dimension: int
    ) -> Basis:
        return basis_factory(input_dimension).build()

else:

    def create_trend_basis(  # noqa: D103
        basis_factory: type(BasisFactory), input_dimension: int, output_dimension: int
    ) -> Basis:
        basis = basis_factory(input_dimension).build()
        return Basis([
            AggregatedFunction([basis.build(k)] * output_dimension)
            for k in range(basis.getSize())
        ])

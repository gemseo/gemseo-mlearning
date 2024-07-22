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

from numpy import array
from numpy.testing import assert_equal

from gemseo_mlearning.active_learning.acquisition_criteria.maximum.output import (  # noqa: E501
    Output,
)


def test_output(algo_distribution):
    """Check the Output criterion."""
    value = array([0.5])
    criterion = Output(algo_distribution)
    mean = algo_distribution.compute_mean(value)
    output_range = criterion.output_range
    assert_equal(criterion.evaluate(value), mean / output_range)

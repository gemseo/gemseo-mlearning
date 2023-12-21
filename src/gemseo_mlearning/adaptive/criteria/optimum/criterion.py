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
r"""Common expected improvement of the regression model.

This is the same as the expected improvement of the regression model for the minimum.

Statistics:

.. math::

   EI[x] = E[\max(y_{min}-Y(x),0)]

where :math:`y_{min}=\min_{1\leq i \leq n}~y^{(i)}`.

Bootstrap estimator:

.. math::

   \widehat{EI}[x] = \frac{1}{B}\sum_{b=1}^B \max(f_{min}-Y_b(x),0)
"""

from __future__ import annotations

from gemseo_mlearning.adaptive.criteria.optimum.criterion_min import (
    MinExpectedImprovement,
)


class ExpectedImprovement(MinExpectedImprovement):
    """The expected improvement.

    This criterion is scaled by the output range.
    """

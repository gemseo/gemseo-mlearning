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
r"""Criterion to be maximized for estimating a minimum, with exploration.

This is the same as the expected improvement of the regression model for the minimum.

Statistic:

$$EI[x] = \mathbb{E}[\max(y_{\text{min}}-Y(x),0)]$$

where $y_{\text{min}}=\min_{1\leq i \leq n}~y^{(i)}$.

Bootstrap estimator:

$$\widehat{EI}[x] = \frac{1}{B}\sum_{b=1}^B \max(y_{\text{min}}-Y_b(x),0)$$
"""

from __future__ import annotations

from gemseo_mlearning.active_learning.acquisition_criteria.min_expected_improvement import (  # noqa: E501
    MinExpectedImprovement,
)


class ExpectedImprovement(MinExpectedImprovement):
    """Criterion to be maximized for estimating a minimum, with exploration.

    This criterion is scaled by the output range.
    """

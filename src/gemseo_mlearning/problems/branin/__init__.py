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
r"""The Branin use case to benchmark and illustrate active learning algorithms.

The Branin function
$$f(x_1,x_2) =
\left(15x_2 - \frac{5.1}{4\pi^2}(15x_1-5)^2 + \frac{5}{\pi}(15x_1-5)-6\right)^2
+ \left(10- \frac{10}{8\pi}\right)\cos(15x_1-5)
+10$$
is commonly studied through the random variable $Y=f(X_1,X_2)$
where $X_1$ and $X_2$ are independent random variables
uniformly distributed over $[0,1]$.

See [@molga2005test].
"""

from __future__ import annotations

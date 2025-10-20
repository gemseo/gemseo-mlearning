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
"""Constraint strategy."""

from __future__ import annotations

from enum import auto

from strenum import StrEnum


class ConstraintStrategy(StrEnum):
    """The constraint strategy for the EGObox_Egor algorithm.

    Hereafter, AC denotes the acquisition criterion related to the objective.

    The functions computing the outputs to be constrained
    are replaced by Gaussian process models.
    """

    INFILL = auto()
    """Optimize the AC multiplied by the probability of feasibility of constraints."""

    MC = auto()
    """Optimize the AC under constraints using GP means."""

    UTB = auto()
    """Optimize the AC under constraints using GP upper bounds."""

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
"""The settings for the SMT's regression models."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.mlearning.regression.algos.base_regressor_settings import (
    BaseRegressorSettings,
)
from pydantic import Field
from pydantic import field_validator
from smt.surrogate_models.surrogate_model import SurrogateModel
from strenum import StrEnum

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gemseo import StrKeyMapping


def _get_subclasses(cls: type[SurrogateModel]) -> Iterator[type[SurrogateModel]]:
    """Return the subclasses of a class of interest recursively.

    Args:
        cls: The class of interest.

    Yields:
        The subclasses of the class of interest.
    """
    for subclass in cls.__subclasses__():
        yield from _get_subclasses(subclass)
        yield subclass


_NAMES_TO_CLASSES: Final[dict[str, type[SurrogateModel]]] = {
    cls.__name__: cls for cls in _get_subclasses(SurrogateModel)
}
"""The model class names bound to the model classes."""

SMTSurrogateModel = StrEnum("SurrogateModel", list(_NAMES_TO_CLASSES.keys()))
"""The class name of an SMT surrogate model."""


class SMTRegressorSettings(BaseRegressorSettings):
    """The settings for the SMT's regression models."""

    model_class_name: SMTSurrogateModel = Field(
        default=...,
        description=(
            "The class name of a surrogate model available in SMT,"
            "i.e. a subclass of`smt.surrogate_models.surrogate_model.SurrogateModel`."
        ),
    )

    @field_validator("parameters")
    @classmethod
    def __set_print_global(cls, parameters: StrKeyMapping) -> StrKeyMapping:
        """Set `print_global` if missing."""
        new_parameters = {"print_global": False}
        new_parameters.update(parameters)
        return new_parameters

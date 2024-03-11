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
"""A regression model from SMT."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Final
from typing import NoReturn

from gemseo.mlearning.regression.rbf import RBFRegressor
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from numpy import double
from smt.surrogate_models.surrogate_model import SurrogateModel
from strenum import StrEnum

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.core.ml_algo import TransformerType
    from gemseo.typing import NumberArray

_NAMES_TO_CLASSES: Final[dict[str, type[SurrogateModel]]] = {
    cls.__name__: cls for cls in SurrogateModel.__subclasses__()
}

SMTSurrogateModel = StrEnum("SurrogateModel", list(_NAMES_TO_CLASSES.keys()))
"""The class name of an SMT surrogate model."""


class SMTRegressionModel(MLRegressionAlgo):
    """A regression model from SMT.

    !!! note

        [SMT](https://smt.readthedocs.io/) is an open-source Python package
        consisting of libraries of surrogate modeling methods,
        sampling methods, and benchmarking problems.
        [Read this page](https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models.html)
        for the list of surrogate models and options.
    """

    __smt_model: SurrogateModel
    """The SMT surrogate model."""

    def __init__(  # noqa: D107
        self,
        data: IODataset,
        model_class_name: SMTSurrogateModel,
        transformer: TransformerType = MLRegressionAlgo.IDENTITY,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
        **model_options: Any,
    ) -> None:
        """
        Args:
            model_class_name: The class name of a surrogate model available in SMT,
                i.e. a subclass of
                ``smt.surrogate_models.surrogate_model.SurrogateModel``.
            **model_options: The options of the surrogate model.
        """  # noqa: D205 D212 D415
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            function=RBFRegressor.Function.THIN_PLATE,
            **model_options,
        )
        _model_options = {"print_global": False}
        _model_options.update(model_options)
        self.__smt_model = _NAMES_TO_CLASSES[model_class_name](**_model_options)

    def _fit(self, input_data: NumberArray, output_data: NumberArray) -> NoReturn:
        self.__smt_model.set_training_values(input_data, output_data)
        self.__smt_model.train()

    def _predict(self, input_data: NumberArray) -> NoReturn:
        return self.__smt_model.predict_values(input_data.astype(double))

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

from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.mlearning.regression.algos.rbf import RBFRegressor
from numpy import concatenate
from numpy import double
from numpy import newaxis
from smt.surrogate_models.surrogate_model import SurrogateModel
from strenum import StrEnum

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator

    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.core.algos.ml_algo import TransformerType
    from gemseo.typing import NumberArray


def _get_subclasses(cls: type) -> Iterator[type]:
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

SMTSurrogateModel = StrEnum("SurrogateModel", list(_NAMES_TO_CLASSES.keys()))
"""The class name of an SMT surrogate model."""


class SMTRegressor(BaseRegressor):
    """A regression model from SMT.

    !!! note

        [SMT](https://smt.readthedocs.io/) is an open-source Python package
        consisting of libraries of surrogate modeling methods,
        sampling methods, and benchmarking problems.
        [Read this page](https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models.html)
        for the list of surrogate models and options.
    """

    def __init__(
        self,
        data: IODataset,
        model_class_name: SMTSurrogateModel,
        transformer: TransformerType = BaseRegressor.IDENTITY,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
        **model_options: Any,
    ) -> None:
        """
        Args:
            model_class_name: The class name of a surrogate model available in SMT,
                i.e. a subclass of
                `smt.surrogate_models.surrogate_model.SurrogateModel`.
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
        self.algo = _NAMES_TO_CLASSES[model_class_name](**_model_options)

    def _fit(self, input_data: NumberArray, output_data: NumberArray) -> None:
        """
        Raises:
            ValueError: When the training dataset does not include gradient samples
                in the case of a gradient-enhanced surrogate model.
        """  # noqa: D205 D212 D415
        self.algo.set_training_values(input_data, output_data)
        surrogate_model_class_name = self.algo.__class__.__name__
        if not surrogate_model_class_name.startswith("GE"):
            self.algo.train()
            return

        # Gradient-enhanced (GE) surrogate models learn both outputs and gradients.
        try:
            get_view = self.learning_set.get_view
            gradient_group_name = self.learning_set.GRADIENT_GROUP
            jac_data = get_view(group_names=gradient_group_name).to_numpy()
        except KeyError as err:
            msg = (
                f"{surrogate_model_class_name} did not found gradient samples "
                "in the training dataset."
            )
            raise ValueError(msg) from err

        for i in range(self.input_dimension):
            self.algo.set_training_derivatives(
                input_data, jac_data[:, i :: self.input_dimension], i
            )

        self.algo.train()

    def _predict(self, input_data: NumberArray) -> NumberArray:
        return self.algo.predict_values(input_data.astype(double))

    def _predict_jacobian(self, input_data: NumberArray) -> NumberArray:
        return concatenate(
            tuple(
                self.algo.predict_derivatives(input_data, i)[..., newaxis]
                for i in range(input_data.shape[1])
            ),
            axis=2,
        )

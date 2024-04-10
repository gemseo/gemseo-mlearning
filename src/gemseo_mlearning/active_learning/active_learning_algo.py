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
#
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Active learning algorithm using regression algorithms and acquisition criteria."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.algos._progress_bars.custom_tqdm_progress_bar import CustomTqdmProgressBar
from gemseo.algos.database import Database
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.datasets.io_dataset import IODataset
from gemseo.utils.logging_tools import LoggingContext
from numpy import array
from numpy import hstack
from numpy import newaxis
from pandas import concat

from gemseo_mlearning.active_learning.acquisition_criteria.acquisition_criterion_factory import (  # noqa: E501
    AcquisitionCriterionFactory,
)

if TYPE_CHECKING:
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.opt.optimization_library import OptimizationLibrary
    from gemseo.core.discipline import MDODiscipline
    from gemseo.mlearning.core.ml_algo import DataType

    from gemseo_mlearning.active_learning.distributions.base_regressor_distribution import (  # noqa: E501
        BaseRegressorDistribution,
    )

LOGGER = logging.getLogger(__name__)

_CRITERION_FACTORY = AcquisitionCriterionFactory()


class ActiveLearningAlgo:
    """An active learning algorithm."""

    __acquisition_algo: OptimizationLibrary
    """The algorithm to find the new training point(s)."""

    __acquisition_algo_options: dict[str, Any]
    """The options of the algorithm to find the new training points()."""

    __acquisition_criterion: str
    """The name of a data acquisition criterion to find the new training point(s)."""

    __acquisition_criterion_options: dict[str, Any]
    """The option of the data acquisition criterion."""

    __acquisition_problem: OptimizationProblem
    """The acquisition problem."""

    __database: Database
    """The concatenation of the optimization histories related to the new points."""

    __distribution: BaseRegressorDistribution
    """The distribution of the machine learning algorithm."""

    __input_space: DesignSpace
    """The input space on which to look for the new learning point."""

    default_algo_name: ClassVar[str] = "NLOPT_COBYLA"
    """The name of the default algorithm to find the new training point(s).

    Typically a DoE or an optimizer.
    """

    default_doe_options: ClassVar[dict[str, Any]] = {"n_samples": 100}
    """The names and values of the default DoE options."""

    default_opt_options: ClassVar[dict[str, Any]] = {"max_iter": 100}
    """The names and values of the default optimization options."""

    def __init__(
        self,
        criterion: str,
        input_space: DesignSpace,
        distribution: BaseRegressorDistribution,
        **criterion_options: Any,
    ) -> None:
        """
        Args:
            criterion: The name of a data acquisition criterion
                selecting new point(s) to reach a particular goal
                (name of a class inheriting from
                [BaseAcquisitionCriterion][gemseo_mlearning.active_learning.acquisition_criteria.base_acquisition_criterion.BaseAcquisitionCriterion]
                ).
            input_space: The input space on which to look for the new learning point.
            distribution: The distribution of the machine learning algorithm.
            **criterion_options: The options of the data acquisition criterion.

        Raises:
            NotImplementedError: When the output dimension is greater than 1.
        """  # noqa: D205 D212 D415
        if distribution.output_dimension > 1:
            msg = "ActiveLearningAlgo works only with scalar output."
            raise NotImplementedError(msg)
        self.__acquisition_criterion = criterion
        self.__acquisition_criterion_options = criterion_options.copy()
        self.__distribution = distribution
        self.__input_space = input_space
        self.__acquisition_problem = self.__create_acquisition_problem()
        self.__acquisition_algo = OptimizersFactory().create(self.default_algo_name)
        self.__acquisition_algo_options = self.default_opt_options
        self.__database = Database()

    def __create_acquisition_problem(self) -> OptimizationProblem:
        """Create the acquisition problem.

        An acquisition problem is an optimization problem
        whose objective is a data acquisition criterion (to minimize or maximize)
        and whose design space is the input space of the surrogate model.

        Approximate the Jacobian with finite differences if missing.

        Returns:
            The acquisition problem.
        """
        acquisition_problem = OptimizationProblem(self.__input_space)
        acquisition_problem.objective = _CRITERION_FACTORY.create(
            self.__acquisition_criterion,
            algo_distribution=self.__distribution,
            **self.__acquisition_criterion_options,
        )
        acquisition_problem.objective.name = self.__acquisition_criterion

        if not acquisition_problem.objective.has_jac:
            acquisition_problem.differentiation_method = (
                OptimizationProblem.DifferentiationMethod.FINITE_DIFFERENCES
            )

        if acquisition_problem.objective.MAXIMIZE:
            acquisition_problem.change_objective_sign()

        return acquisition_problem

    def set_acquisition_algorithm(self, algo_name: str, **options: Any) -> None:
        """Set sampling or optimization algorithm.

        Args:
            algo_name: The name of the algorithm to find the learning point(s).
                Typically a DoE or an optimizer.
            **options: The values of some algorithm options;
                use the default values for the other ones.
        """
        factory = DOEFactory()
        if factory.is_available(algo_name):
            self.__acquisition_algo_options = self.default_doe_options.copy()
        else:
            factory = OptimizersFactory()
            self.__acquisition_algo_options = self.default_opt_options.copy()

        self.__acquisition_algo_options.update(options)
        self.__acquisition_algo = factory.create(algo_name)

    def compute_next_input_data(
        self,
        as_dict: bool = False,
    ) -> DataType:
        """Find the next learning point.

        Args:
            as_dict: Whether to return the input data split by input names.
                Otherwise, return a unique array.

        Returns:
            The next learning point.
        """
        with LoggingContext(logging.getLogger("gemseo")):
            input_data = self.__acquisition_algo.execute(
                self.__acquisition_problem, **self.__acquisition_algo_options
            ).x_opt

        if as_dict:
            return self.__acquisition_problem.design_space.array_to_dict(input_data)

        return input_data

    def update_algo(
        self, discipline: MDODiscipline, n_samples: int = 1
    ) -> tuple[Database, OptimizationProblem]:
        """Update the machine learning algorithm by learning new samples.

        This method acquires new learning input-output samples
        and trains the machine learning algorithm
        with the resulting enriched learning set.

        Args:
            discipline: The discipline computing the reference output data
                from the input data provided by the acquisition process.
            n_samples: The number of samples to update the machine learning algorithm.

        Returns:
            The concatenation of the optimization histories
            related to the different points
            and the last acquisition problem.
        """
        LOGGER.info("Update machine learning algorithm with %s points", n_samples)
        for sample_id in CustomTqdmProgressBar(range(1, n_samples + 1)):
            input_data = self.compute_next_input_data(as_dict=True)
            for inputs, outputs in self.__acquisition_problem.database.items():
                self.__database.store(
                    array([sample_id, *inputs.unwrap().tolist()]), outputs
                )

            discipline.execute(input_data)

            extra_learning_set = IODataset()
            distribution = self.__distribution
            variable_names_to_n_components = distribution.algo.sizes
            extra_learning_set.add_group(
                group_name=IODataset.INPUT_GROUP,
                data=hstack(list(input_data.values()))[newaxis],
                variable_names=distribution.input_names,
                variable_names_to_n_components=variable_names_to_n_components,
            )
            output_names = distribution.output_names
            extra_learning_set.add_group(
                group_name=IODataset.OUTPUT_GROUP,
                data=hstack([
                    discipline.local_data[output_name] for output_name in output_names
                ])[newaxis],
                variable_names=output_names,
                variable_names_to_n_components=variable_names_to_n_components,
            )
            augmented_learning_set = concat(
                [distribution.algo.learning_set, extra_learning_set],
                ignore_index=True,
            )

            self.__distribution.change_learning_set(augmented_learning_set)
            self.update_problem()

        return self.__database, self.__acquisition_problem

    def update_problem(self) -> None:
        """Update the acquisition problem."""
        self.__acquisition_problem = self.__create_acquisition_problem()

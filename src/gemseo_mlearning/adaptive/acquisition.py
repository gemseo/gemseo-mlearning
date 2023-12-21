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
"""Acquisition of learning data from a machine learning algorithm and a criterion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.algos.database import Database
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.datasets.io_dataset import IODataset
from numpy import array
from pandas import concat

from gemseo_mlearning.adaptive.criterion import MLDataAcquisitionCriterionFactory
from gemseo_mlearning.adaptive.criterion import MLDataAcquisitionCriterionOptionType

if TYPE_CHECKING:
    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.discipline import MDODiscipline
    from gemseo.mlearning.core.ml_algo import DataType

    from gemseo_mlearning.adaptive.distribution import MLRegressorDistribution

LOGGER = logging.getLogger(__name__)

_CRITERION_FACTORY = MLDataAcquisitionCriterionFactory()


class MLDataAcquisition:
    """Data acquisition for adaptive learning."""

    default_algo_name: ClassVar[str] = "NLOPT_COBYLA"
    """The name of the default algorithm to find the point(s).

    Typically a DoE or an optimizer.
    """
    default_opt_options: ClassVar[dict[str, Any]] = {"max_iter": 100}
    """The names and values of the default optimization options."""
    default_doe_options: ClassVar[dict[str, Any]] = {"n_samples": 100}
    """The names and values of the default DoE options."""

    def __init__(
        self,
        criterion: str,
        input_space: DesignSpace,
        distribution: MLRegressorDistribution,
        **options: MLDataAcquisitionCriterionOptionType,
    ) -> None:
        """
        Args:
            criterion: The name of a data acquisition criterion
                selecting new point(s) to reach a particular goal
                (name of a class inheriting from :class:`.MLDataAcquisitionCriterion`).
            input_space: The input space on which to look for the new learning point.
            distribution: The distribution of the machine learning algorithm.
            **options: The options of the acquisition criterion.

        Raises:
            NotImplementedError: When the output dimension is greater than 1.
        """  # noqa: D205 D212 D415
        if distribution.output_dimension > 1:
            raise NotImplementedError(
                "MLDataAcquisition works only with scalar output."
            )
        self.__algo_name = self.default_algo_name
        self.__algo_options = self.default_opt_options
        self.__algo = OptimizersFactory().create(self.__algo_name)
        self.__criterion = criterion
        self.__input_space = input_space
        self.__criterion_options = options.copy()
        self.__distribution = distribution
        self.__database = Database()
        self.__problem = self.__build_optimization_problem()

    def __build_optimization_problem(self) -> OptimizationProblem:
        """Create the optimization problem.

        The data acquisition criterion is the objective (either a cost or a performance)
        while the input space is the design space.

        Approximate the Jacobian with finite differences if missing.

        Returns:
            The optimization problem.
        """
        problem = OptimizationProblem(self.__input_space)
        problem.objective = _CRITERION_FACTORY.create(
            self.__criterion,
            algo_distribution=self.__distribution,
            **self.__criterion_options,
        )
        problem.objective.name = self.__criterion

        if not problem.objective.has_jac:
            problem.differentiation_method = (
                OptimizationProblem.DifferentiationMethod.FINITE_DIFFERENCES
            )

        if problem.objective.MAXIMIZE:
            problem.change_objective_sign()

        return problem

    def set_acquisition_algorithm(self, algo_name: str, **options: Any) -> None:
        """Set sampling or optimization algorithm.

        Args:
            algo_name: The name of the algorithm to find the learning point(s).
                Typically a DoE or an optimizer.
            **options: The values of some algorithm options;
                use the default values for the other ones.
        """
        self.__algo_name = algo_name
        factory = DOEFactory()
        if factory.is_available(algo_name):
            self.__algo_options = self.default_doe_options.copy()
        else:
            factory = OptimizersFactory()
            self.__algo_options = self.default_opt_options.copy()

        self.__algo_options.update(options)
        self.__algo = factory.create(algo_name)

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
        input_data = self.__algo.execute(self.__problem, **self.__algo_options).x_opt
        if as_dict:
            return self.__problem.design_space.array_to_dict(input_data)

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
            and the last optimization problem.
        """
        for index in range(n_samples):
            root_logger = logging.getLogger()
            saved_level = root_logger.level
            root_logger.setLevel(logging.WARNING)
            LOGGER.setLevel(logging.WARNING)
            input_data = self.compute_next_input_data(as_dict=True)
            for inputs, outputs in self.__problem.database.items():
                self.__database.store(
                    array([index + 1, *inputs.unwrap().tolist()]), outputs
                )

            discipline.execute(input_data)

            # TODO: This is a dirty fix. Please refactor this function.
            # WARNING: Using concat like this could lead to performance issues.
            dataset_to_add = IODataset()
            for input_name, input_value in input_data.items():
                dataset_to_add.add_variable(
                    input_name, input_value, group_name=dataset_to_add.INPUT_GROUP
                )
            for output_name in self.__distribution.output_names:
                dataset_to_add.add_variable(
                    output_name,
                    discipline.local_data[output_name],
                    group_name=dataset_to_add.OUTPUT_GROUP,
                )
            dataset = concat(
                [self.__distribution.algo.learning_set, dataset_to_add],
                ignore_index=True,
            )

            self.__distribution.change_learning_set(dataset)
            self.update_problem()
            LOGGER.setLevel(saved_level)
            LOGGER.info("Add sample %s out of %s", index + 1, n_samples)

        return self.__database, self.__problem

    def update_problem(self) -> None:
        """Update the optimization problem."""
        self.__problem = self.__build_optimization_problem()

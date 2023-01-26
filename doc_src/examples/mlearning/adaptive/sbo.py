"""Surrogate-based optimization of Rosenbrock's function."""
from __future__ import annotations

from typing import Any

from gemseo.post.dataset.scatter import Scatter
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo_mlearning.adaptive.acquisition import MLDataAcquisition
from gemseo_mlearning.adaptive.distributions.regressor_distribution import \
    RegressorDistribution
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.doe.doe_lib import DOELibraryOptionType
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.mlearning.regression.factory import RegressionModelFactory


class SBO:
    """Surrogate-based optimization."""

    @staticmethod
    def execute(
        problem: OptimizationProblem,
        doe_algorithm: str,
        doe_options: dict[str, DOELibraryOptionType],
        model_name: str,
        model_options: dict[str, Any],
        criterion_name: str,
        criterion_options: dict[str, Any],
        acquisition_algorithm: str,
        acquisition_options: dict[str, Any],
        number_of_acquisitions: int
    ):
        # Generate the initial sample
        DOEFactory().execute(problem, doe_algorithm, **doe_options)

        # Initialize the surrogate model of the objective function
        model = RegressionModelFactory().create(
            model_name,
            data=problem.export_to_dataset(opt_naming=False),
            **model_options
        )

        # Initialize the data acquisition
        distribution = RegressorDistribution(model, bootstrap=False)
        distribution.learn()
        acquisition = MLDataAcquisition(
            criterion_name, problem.design_space, distribution, **criterion_options
        )
        acquisition.set_acquisition_algorithm(
            acquisition_algorithm, **acquisition_options
        )

        # Acquire data
        problem.max_iter = len(problem.database) + number_of_acquisitions
        for _ in range(number_of_acquisitions):
            problem.evaluate_functions(
                acquisition.compute_next_input_data(), normalize=False
            )
            distribution.change_learning_set(
                problem.export_to_dataset(opt_naming=False)
            )
            acquisition.update_problem()


if __name__ == "__main__":

    problem = Rosenbrock()

    SBO().execute(
        problem=problem,
        doe_algorithm="OT_OPT_LHS",
        doe_options={"n_samples": 30},
        model_name="RBFRegressor",
        model_options={},
        criterion_name="ExpectedImprovement",
        criterion_options={},
        acquisition_algorithm="OT_OPT_LHS",
        acquisition_options={"n_samples": 100},
        number_of_acquisitions=20
    )
    f, x, _, _, _ = problem.get_optimum()

    print(f"Solution: {f} {x}")

    PostFactory().execute(problem, "OptHistoryView", show=True)
    Scatter(problem.export_to_dataset(), "x", "x", 0, 1).execute(save=False, show=True)

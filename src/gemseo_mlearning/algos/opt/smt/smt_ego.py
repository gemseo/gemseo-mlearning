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
#    AUTHORS:
#       - Francois Gallard
"""The efficient global optimization (EGO) algorithm of SMT."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from gemseo.algos.design_space_utils import get_value_and_bounds
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.optimization_result import OptimizationResult
from numpy import atleast_2d
from smt.applications.ego import EGO
from smt.surrogate_models import GEKPLS
from smt.surrogate_models import GPX
from smt.surrogate_models import KPLS
from smt.surrogate_models import KPLSK
from smt.surrogate_models import KRG
from smt.surrogate_models import MGP
from smt.utils.design_space import DesignSpace as SMTDesignSpace

from gemseo_mlearning.algos.opt.smt._parallel_evaluator import ParallelEvaluator
from gemseo_mlearning.algos.opt.smt.ego_settings import SMT_EGO_Settings

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.typing import RealArray
    from smt.surrogate_models.surrogate_model import SurrogateModel


class SMTEGO(BaseOptimizationLibrary):
    """Surrogate-based optimizers from SMT."""

    __NAMES_TO_CLASSES: Final[dict[str, type[SurrogateModel]]] = {
        "GEKPLS": GEKPLS,
        "GPX": GPX,
        "KPLS": KPLS,
        "KPLSK": KPLSK,
        "KRG": KRG,
        "MGP": MGP,
    }
    """The names of the surrogate models bound to the SMT classes."""

    ALGORITHM_INFOS: ClassVar[dict[str, OptimizationAlgorithmDescription]] = {
        "SMT_EGO": OptimizationAlgorithmDescription(
            library_name="SMT",
            algorithm_name="SMT_EGO",
            description="Efficient Global Optimization",
            internal_algorithm_name="SMT_EGO",
            website="https://smt.readthedocs.io/en/latest/_src_docs/applications/ego.html",  # noqa: E501
            Settings=SMT_EGO_Settings,
        )
    }

    def __init__(self, algo_name: str = "SMT_EGO") -> None:  # noqa: D107
        super().__init__(algo_name)

    def _run(
        self, problem: OptimizationProblem, **settings: Any
    ) -> tuple[None, None, RealArray, RealArray, RealArray]:
        design_space = problem.design_space
        x_0, lower_bounds, upper_bounds = get_value_and_bounds(
            design_space, normalize_ds=self._normalize_ds
        )
        surrogate = settings["surrogate"]
        if isinstance(surrogate, str):
            smt_design_space = SMTDesignSpace(
                atleast_2d(
                    list(
                        zip(
                            [
                                lower_bound if isfinite(lower_bound) else None
                                for lower_bound in lower_bounds
                            ],
                            [
                                upper_bound if isfinite(upper_bound) else None
                                for upper_bound in upper_bounds
                            ],
                        )
                    )
                )
            )
            surrogate = self.__NAMES_TO_CLASSES[surrogate](
                design_space=smt_design_space, print_global=False
            )

        n_parallel = settings["n_parallel"]
        evaluator = ParallelEvaluator(n_parallel)

        max_iter = settings["max_iter"]
        n_doe = settings["n_doe"]
        n_iter = max_iter - n_doe - 1
        if n_iter <= 0:
            msg = (
                f"max_iter must be strictly greater than n_doe+1 = {n_doe + 1}; "
                f"got {max_iter}."
            )
            raise ValueError(msg)

        ego = EGO(
            surrogate=surrogate,
            n_doe=n_doe,
            n_iter=n_iter,
            criterion=settings["criterion"],
            n_parallel=n_parallel,
            qEI=settings["qEI"],
            n_start=settings["n_start"],
            evaluator=evaluator,
            random_state=settings["random_state"],
            n_max_optim=settings["n_max_optim"],
        )
        x_opt, y_opt, _, _, _ = ego.optimize(fun=problem.objective.evaluate)
        if self._normalize_ds:
            x_opt = design_space.unnormalize_vect(x_opt)
            x_0 = design_space.unnormalize_vect(x_0)
        return (
            None,
            None,
            x_0,
            x_opt,
            y_opt,
        )

    def _get_result(
        self,
        problem: OptimizationProblem,
        message: Any,
        status: Any,
        x_0: RealArray,
        x_opt: RealArray,
        f_opt: RealArray,
    ) -> OptimizationResult:
        return OptimizationResult(
            x_0=x_0,
            x_opt=x_opt,
            f_opt=f_opt,
            optimizer_name=self.algo_name,
            n_obj_call=problem.objective.n_calls,
            is_feasible=True,
        )

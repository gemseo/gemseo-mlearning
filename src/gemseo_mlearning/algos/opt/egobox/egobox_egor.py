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
"""The efficient global optimization (EGO) algorithm of EGObox."""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import NoReturn

from egobox import ConstraintStrategy
from egobox import CorrelationSpec
from egobox import Egor
from egobox import GpConfig as _GPConfig
from egobox import InfillOptimizer
from egobox import InfillStrategy
from egobox import QInfillStrategy
from egobox import Recombination
from egobox import RegressionSpec
from gemseo import compute_doe
from gemseo.algos.design_space_utils import get_value_and_bounds
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.stop_criteria import MaxIterReachedException
from numpy import array
from numpy import ndarray

from gemseo_mlearning.algos.opt.egobox.constraint_strategy import (
    ConstraintStrategy as _ConstraintStrategy,
)
from gemseo_mlearning.algos.opt.egobox.egor_settings import EGObox_Egor_Settings

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    from gemseo.typing import RealArray


class _EgorFunctions:
    """Functions for the Egor class."""

    def __init__(
        self, objective: MDOFunction, constraints: Iterable[MDOFunction]
    ) -> None:
        """
        Args:
            objective: The objective function.
            constraints: The constraint functions.
        """  # noqa: D205, D212
        self.functions = [objective.evaluate, *(c.evaluate for c in constraints)]

    @staticmethod
    def __convert_value_to_scalar(value: RealArray | float) -> float:
        """Convert a value to scalar.

        Args:
            value: The value to convert.

        Returns:
            The scalar value.
        """
        return value[0] if isinstance(value, ndarray) else value

    def __call__(self, x: RealArray) -> RealArray:
        """Evaluate the functions.

        Args:
            x: The input values, shaped as `(n_samples, input_dimension)`.

        Returns:
            The evaluations of the functions.
        """
        functions = self.functions
        return array([
            [self.__convert_value_to_scalar(f(xi)) for f in functions] for xi in x
        ])


class EGOboxEgor(BaseOptimizationLibrary[EGObox_Egor_Settings]):
    """Efficient global optimization (EGO) using EGObox."""

    ALGORITHM_INFOS: ClassVar[dict[str, OptimizationAlgorithmDescription]] = {
        "EGObox_Egor": OptimizationAlgorithmDescription(
            library_name="EGObox",
            algorithm_name="EGObox_Egor",
            description="Efficient Global Optimization (EGO)",
            internal_algorithm_name="Egor",
            website="https://github.com/relf/egobox/blob/master/python/egobox/egobox.pyi",  # noqa: E501
            Settings=EGObox_Egor_Settings,
            handle_inequality_constraints=True,
        )
    }

    def __init__(self, algo_name: str = "EGObox_Egor") -> None:  # noqa: D107
        super().__init__(algo_name)

    def _run(
        self,
        problem: OptimizationProblem,
    ) -> NoReturn:
        """
        Raises:
            ValueError: When the maximum number of iterations
                is lower than the DOE size plus 2.
            MaxIterReachedException: In the absence of early stopping.
        """  # noqa: D205, D212
        n_initial_samples = self.__create_doe()
        max_iter = self._settings.max_iter - n_initial_samples - 1
        if max_iter <= 0:
            msg = (
                "The maximum number of iterations must be strictly greater "
                f"than 1 + n_initial_samples (={n_initial_samples + 1})."
            )
            raise ValueError(msg)

        design_space = self.__create_design_space()

        inequality_constraints = list(problem.constraints.get_inequality_constraints())
        n_cstr = len(inequality_constraints)
        kwargs = {}
        if self._settings.cstr_strategy != _ConstraintStrategy.INFILL:
            kwargs["cstr_strategy"] = getattr(
                ConstraintStrategy, self._settings.cstr_strategy
            )
        ego = Egor(
            design_space,
            self.__create_gp_config(),
            n_cstr=n_cstr,
            cstr_tol=[self._settings.ineq_tolerance] * n_cstr,
            n_start=self._settings.n_start,
            n_doe=self._settings.n_doe,
            doe=self._settings.doe,
            infill_strategy=getattr(InfillStrategy, self._settings.infill_strategy),
            cstr_infill=self._settings.cstr_strategy == _ConstraintStrategy.INFILL,
            q_points=self._settings.q_points,
            q_infill_strategy=getattr(
                QInfillStrategy, self._settings.q_infill_strategy
            ),
            infill_optimizer=getattr(InfillOptimizer, self._settings.infill_optimizer),
            trego=self._settings.trego,
            coego_n_coop=self._settings.coego_n_coop,
            q_optmod=self._settings.q_optmod,
            target=self._settings.target,
            outdir=self._settings.outdir,
            warm_start=self._settings.warm_start,
            hot_start=self._settings.hot_start,
            seed=self._settings.seed,
            **kwargs,
        )
        ego.minimize(
            _EgorFunctions(problem.objective, inequality_constraints),
            max_iters=max_iter,
        )
        raise MaxIterReachedException

    def __create_doe(self) -> int:
        """Create the DOE if not passed.

        Returns:
            The initial number of samples.
        """
        if self._settings.doe is None:
            n_initial_samples = self._settings.n_doe
        else:
            if isinstance(self._settings.doe, ndarray):
                doe = self._settings.doe
            else:
                doe = compute_doe(
                    self._problem.design_space, settings_model=self._settings.doe
                )

            self._settings.doe = self._problem.design_space.normalize_vect(doe)
            n_initial_samples = len(doe)

        return n_initial_samples

    def __create_design_space(self) -> list[tuple[RealArray, RealArray]]:
        """Create the design space for Egor.

        Returns:
            The design space for Egor.
        """
        _, lower_bounds, upper_bounds = get_value_and_bounds(
            self._problem.design_space,
            normalize_ds=self._settings.normalize_design_space,
        )
        return list(
            zip(
                [
                    lower_bound if isfinite(lower_bound) else None
                    for lower_bound in lower_bounds
                ],
                [
                    upper_bound if isfinite(upper_bound) else None
                    for upper_bound in upper_bounds
                ],
                strict=False,
            )
        )

    def __create_gp_config(self) -> _GPConfig:
        """Create the GP configuration for Egor.

        Returns:
            The GP configuration.
        """
        gp_settings = self._settings.gp_config.model_dump()
        gp_settings["recombination"] = getattr(
            Recombination, "SMOOTH" if gp_settings.pop("use_smooth_moe") else "HARD"
        )
        gp_settings["regr_spec"] = getattr(RegressionSpec, gp_settings["regr_spec"])
        gp_settings["corr_spec"] = getattr(CorrelationSpec, gp_settings["corr_spec"])

        theta_init = gp_settings["theta_init"]
        default_theta_init = gp_settings.pop("default_theta_init")
        gp_settings["theta_init"] = [
            v
            for name, size in self._problem.design_space.variable_sizes.items()
            for v in theta_init.get(name, [default_theta_init] * size)
        ]

        theta_bounds = gp_settings["theta_bounds"]
        default_l_b, default_u_b = gp_settings.pop("default_theta_bounds")
        gp_settings["theta_bounds"] = [
            list(v)
            for name, size in self._problem.design_space.variable_sizes.items()
            for v in zip(
                *theta_bounds.get(name, ([default_l_b] * size, [default_u_b] * size)),
                strict=False,
            )
        ]

        return _GPConfig(**gp_settings)

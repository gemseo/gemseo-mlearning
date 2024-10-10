# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""# Surrogate-based optimization of Rastrigin's function."""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import execute_algo
from gemseo import execute_post
from gemseo.problems.optimization.rastrigin import Rastrigin

configure_logger()

problem = Rastrigin()

execute_algo(
    problem,
    algo_name="SBO",
    acquisition_settings={
        "max_iter": 1000,
        "popsize": 50,
        "stop_crit_n_x": 10000,
    },
    max_iter=20,
)

execute_post(problem, "OptHistoryView", save=False, show=True)

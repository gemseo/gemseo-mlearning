<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->
# gemseo-mlearning

[![PyPI - License](https://img.shields.io/pypi/l/gemseo-mlearning)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gemseo-mlearning)](https://pypi.org/project/gemseo-mlearning/)
[![PyPI](https://img.shields.io/pypi/v/gemseo-mlearning)](https://pypi.org/project/gemseo-mlearning/)
[![Codecov branch](https://img.shields.io/codecov/c/gitlab/gemseo:dev/gemseo-mlearning/develop)](https://app.codecov.io/gl/gemseo:dev/gemseo-mlearning)

`gemseo-mlearning` is a plugin of the library [GEMSEO](https://www.gemseo.org), dedicated to machine learning.
This package is open-source,
under the [LGPL v3 license](https://www.gnu.org/licenses/lgpl-3.0.en.html).

## Overview

This package adds new [regression models][gemseo_mlearning.regression]
and [optimization algorithms][gemseo_mlearning.algos.opt.smt]
based on [SMT](https://smt.readthedocs.io/).

A [package for active learning][gemseo_mlearning.active_learning] is also available,
deeply based on the core GEMSEO objects for optimization,
as well as a
[SurrogateBasedOptimization][gemseo_mlearning.algos.opt.surrogate_based_optimization.SurrogateBasedOptimization]
library built on its top.
An effort is being made to improve both content and performance in future versions.

## Installation

Install the latest version with `pip install gemseo-mlearning`.

See [pip](https://pip.pypa.io/en/stable/getting-started/) for more information.

## Bugs and questions

Please use the [gitlab issue tracker](https://gitlab.com/gemseo/dev/gemseo-mlearning/-/issues)
to submit bugs or questions.

## Contributing

See the [contributing section of GEMSEO](https://gemseo.readthedocs.io/en/stable/software/developing.html#dev).

## Contributors

- Antoine Dechaume
- Benoît Pauwels
- Clément Laboulfie
- Matthias De Lozzo

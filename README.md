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

## Overview

`gemseo-mlearning` is a plugin of the library [GEMSEO](https://www.gemseo.org), dedicated to machine learning.

This package adds new [regression models](https://gemseo.gitlab.io/dev/gemseo-mlearning/latest/user_guide/optimization/smt/)
and [optimization algorithms](https://gemseo.gitlab.io/dev/gemseo-mlearning/latest/user_guide/regression/smt/)
based on [SMT](https://smt.readthedocs.io/).

A [package for active learning](https://gemseo.gitlab.io/dev/gemseo-mlearning/latest/user_guide/active_learning/what_active_learning_is/) is also available,
deeply based on the core GEMSEO objects for optimization,
as well as a
[SurrogateBasedOptimization](https://gemseo.gitlab.io/dev/gemseo-mlearning/latest/user_guide/optimization/al/)
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

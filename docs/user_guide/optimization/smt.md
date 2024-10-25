<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->
# SMT's surrogate-based optimizers

The [surrogate modeling toolbox (SMT)](https://smt.readthedocs.io)
is an open-source Python package for surrogate modeling with a focus on derivatives [@SMT2019,@saves2024smt].

`gemseo-mlearning` proposes the [SMTEGO][gemseo_mlearning.algos.opt.smt.smt_ego.SMTEGO] optimization library
to easily use the surrogate-based optimizers available in SMT,
through its `EGO` class.

## Basic usage

[SMTEGO][gemseo_mlearning.algos.opt.smt.smt_ego.SMTEGO] includes a single optimization algorithm,
called `"SMT_EGO"`.

Given a maximum number of iterations,
it can be used as is
by any [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]:
```python
execute_algo(optimization_problem, "SMT_EGO", max_iter=50)
```
and any [MDOScenario][gemseo.scenarios.mdo_scenario.MDOScenario]:
```python
scenario.execute({"algo_name": "SMT_EGO", "max_iter": 50})
```

In this case,
the settings are

- the expected improvement as acquisition criterion,
- 1 point acquired at a time,
- the Kriging-based surrogate model `"KRG"`,
- 10 initial training points based on a latin hypercube sampling (LHS) technique,
- a multi-start local optimization of the acquisition criterion
  from 50 start points with a limit of 20 iterations per local optimization.

## Settings

Regarding the settings of the SMT's `EGO` class,
you will find more information in the [SMT's user guide](https://smt.readthedocs.io/en/latest/_src_docs/applications/ego.html)
by looking at the table at the bottom of the page.

In the following,
the training output values already used
and the output and uncertainty predictions at a given input point $x$
are respectively denoted $\{y_1,\ldots,y_n\}$, $\mu(x)$ and $\sigma(x)$.

### General

::: gemseo_mlearning.algos.opt.smt.ego_settings.SMTEGOSettings
    options:
      show_root_heading: false
      show_bases: false
      show_root_toc_entry: false

### Acquisition criteria

The three acquisition criteria are

| Value   | Name                   | Expression                                 |
|---------|------------------------|--------------------------------------------|
| `"EI"`  | Expected improvement   | $\mathbb{E}[\max(\min(y_1,\dots,y_n)-Y,0]$ |
| `"LCB"` | Lower confidence bound | $\mu(x)-3\times\sigma(x)$                  |
| `"SBO"` | Kriging believer       | $\mu(x)$                                   |

where $Y$ is a Gaussian random variable with mean $\mu(x)$ and standard deviation $\sigma(x)$,

### Parallel acquisition

Points can be acquired by batch of $q>1$ points,
as Kriging is well-suited to parallelize optimization[@ginsbourger2010kriging].
To this aim,
when `n_parallel` is greater than 1,
SMT uses a technique of virtual points to update the training dataset with $k\leq q$ training points
whose output value mimics the substituted model using a strategy.
The four strategies are:

| Value      | Name                          | Expression                  |
|------------|-------------------------------|-----------------------------|
| `"CLmin"`  | Minimum constant liar         | $\min \{y_1,\ldots,y_n\}$   |
| `"KB"`     | Kriging believer              | $\mu(x)$                    |
| `"KBLB"`   | Kriging believer lower bound  | $\mu(x)-3\sigma(x)$         |
| `"KBRand"` | Kriging believer random bound | $\mu(x)+\kappa(x)\sigma(x)$ |
| `"KBUB"`   | Kriging believer upper bound  | $\mu(x)+3\sigma(x)$         |

where $\kappa(x)$ is the realization of a random variable distributed according to the standard normal distribution.

### Surrogate models

| Value     | Name                                                                                  |
|-----------|---------------------------------------------------------------------------------------|
| `"GPX"`   | Kriging based on the [egobox](https://github.com/relf/egobox) library written in Rust |
| `"KRG"`   | Kriging                                                                               |
| `"KPLS"`  | Kriging using partial least squares (PLS) to reduce the input dimension               |
| `"KPLSK"` | A variant of KPLS                                                                     |
| `"MGP"`   | A marginal Gaussian process (MGP) regressor                                           |

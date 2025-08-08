<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->
# SMT's surrogate-based optimizers

![SMT](../../images/smt_logo.webp){ align=right } The [surrogate modeling toolbox (SMT)](https://smt.readthedocs.io)
is an open-source Python package for surrogate modeling with a focus on derivatives [@SMT2019][@saves2024smt].

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
execute_algo(optimization_problem, algo_name="SMT_EGO", max_iter=50)
```
and any [MDOScenario][gemseo.scenarios.mdo_scenario.MDOScenario]:
```python
scenario.execute(algo_name="SMT_EGO", max_iter=50)
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

This section presents the options of the optimization algorithm `"SMT_EGO"`.

Their default values are defined in [SMT_EGO_Settings][gemseo_mlearning.algos.opt.smt.ego_settings.SMT_EGO_Settings].

!!! info
    You will find more information in the [SMT's user guide](https://smt.readthedocs.io/en/latest/_src_docs/applications/ego.html).

### Acquisition criteria

You can use the option `criterion` to change the acquisition criterion:

| Value   | Name                   | Expression                                 |
|---------|------------------------|--------------------------------------------|
| `"EI"`  | Expected improvement   | $\mathbb{E}[\max(\min(y_1,\dots,y_n)-Y,0]$ |
| `"LCB"` | Lower confidence bound | $\mu(x)-3\times\sigma(x)$                  |
| `"SBO"` | Kriging believer       | $\mu(x)$                                   |

where $Y$ is a Gaussian random variable with mean function $\mu$ and standard deviation function $\sigma$,
and where $\{y_1,\ldots,y_n\}$ denote the training output values already used.

### Optimization algorithm

The optimization algorithm `"SMT_EGO"` uses sub-optimizations to optimize the acquisition criterion.
The number of sub-optimizations is parametrized by `n_start`
while the maximum number of iterations for each sub-optimization is parametrized by `n_max_optim`.

### Parallel acquisition


Points can be acquired by batch of $q>1$ points,
as Kriging is well-suited to parallelize optimization[@ginsbourger2010kriging].
To this aim,
when `n_parallel` is greater than 1,
SMT uses a technique of virtual points to update the training dataset with $k\leq q$ training points
whose output value mimics the substituted model using a strategy.

You can use the options `n_parallel` to acquire `n_parallel` points in parallel using the acquisition strategy `qEI`.

The strategies are:

| Value      | Name                          | Expression                  |
|------------|-------------------------------|-----------------------------|
| `"CLmin"`  | Minimum constant liar         | $\min \{y_1,\ldots,y_n\}$   |
| `"KB"`     | Kriging believer              | $\mu(x)$                    |
| `"KBLB"`   | Kriging believer lower bound  | $\mu(x)-3\sigma(x)$         |
| `"KBRand"` | Kriging believer random bound | $\mu(x)+\kappa(x)\sigma(x)$ |
| `"KBUB"`   | Kriging believer upper bound  | $\mu(x)+3\sigma(x)$         |

where $\kappa(x)$ is the realization of a random variable distributed according to the standard normal distribution.

You can also enable the penalization of points that have been already evaluated in EI criterion,
by using the option `enable_tunneling`.

### Surrogate models

You can use the option `surrogate` to change the surrogate model:

| Value     | Name                                                                                  |
|-----------|---------------------------------------------------------------------------------------|
| `"GPX"`   | Kriging based on the [egobox](https://github.com/relf/egobox) library written in Rust |
| `"KRG"`   | Kriging                                                                               |
| `"KPLS"`  | Kriging using partial least squares (PLS) to reduce the input dimension               |
| `"KPLSK"` | A variant of KPLS                                                                     |
| `"MGP"`   | A marginal Gaussian process (MGP) regressor                                           |

You can also change the size of the training dataset using the option `n_doe`.

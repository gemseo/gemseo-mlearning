<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->
# EGObox' surrogate-based optimizers

![EGObox](../../images/egobox.png){ align=right }
The [efficient global optimization (EGO) toolbox EGObox](https://github.com/relf/egobox/tree/master)
is an open-source Rust package for Bayesian optimization[@Lafage2022], with a Python API.

`gemseo-mlearning` proposes the [EGOboxEgor][gemseo_mlearning.algos.opt.egobox.egobox_egor.EGOboxEgor] optimization library
to easily use the surrogate-based optimizers available in EGObox,
through its `Egor` class.

## Basic usage

[EGOboxEgor][gemseo_mlearning.algos.opt.egobox.egobox_egor.EGOboxEgor] includes a single optimization algorithm,
called `"EGObox_Egor"`.

Given a maximum number of iterations,
it can be used as is
by any [OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem]:
```python
execute_algo(optimization_problem, algo_name="EGObox_Egor", max_iter=50)
```
and any [MDOScenario][gemseo.scenarios.mdo_scenario.MDOScenario]:
```python
scenario.execute(algo_name="EGObox_Egor", max_iter=50)
```

In this case,
the settings are

- the WB2 (Watson and Barnes 2nd criterion) as acquisition criterion,
- 1 point acquired at a time,
- a Kriging-based surrogate model,
- 10 initial training points based on a Latin hypercube sampling (LHS) technique,
- a COBYLA-based multi-start local optimization of the acquisition criterion
  from 20 start points with a limit of 50 iterations per local optimization.

## Settings

This section presents the options of the optimization algorithm `"EGObox_Egor"`.

Their default values are defined in
[EGObox_Egor_Settings][gemseo_mlearning.algos.opt.egobox.egor_settings.EGObox_Egor_Settings].

### Acquisition criteria

You can use the option `infill_strategy`  (default: `"WB2"`) to change the acquisition criterion:

| Value      | Name                                          | Expression                                            |
|------------|-----------------------------------------------|-------------------------------------------------------|
| `"EI"`     | Expected improvement                          | $\mathbb{E}[\max(\min(y_1,\dots,y_n)-Y(x),0]$         |
| `"WB2"`    | Watson and Barnes 2nd criterion               | $\mathbb{E}[\max(\min(y_1,\dots,y_n)-Y(x),0]-\mu(x)$  |
| `"WB2S"`   | Watson and Barnes 2nd criterion with scaling  | $s\mathbb{E}[\max(\min(y_1,\dots,y_n)-Y(x),0]-\mu(x)$ |
| `"LOG_EI"` | Logarithm of the expected improvement         | $\log(\mathbb{E}[\max(\min(y_1,\dots,y_n)-Y(x),0])$   |

where $Y$ is a Gaussian random variable with mean function $\mu$ and standard deviation function $\sigma$,
and where $\{y_1,\ldots,y_n\}$ denote the training output values already used.

### Optimization algorithm

The optimization algorithm `"EGObox_Egor"` uses sub-optimizations to maximize the acquisition criterion.
The optimizer parametrized by `infill_optimizer` is either `"COBYLA"` (default) or `"SLSQP"`
and the number of sub-optimizations is parametrized by `n_start` (default: `20`).

### Parallel acquisition

Points can be acquired by batch of $q>1$ points,
as Kriging is well-suited to parallelize optimization[@ginsbourger2010kriging].
To this aim,
when `q_points` is greater than 1,
EGObox uses a technique of virtual points to update the training dataset with training points
whose output value mimics the substituted model using a batch infill strategy.

This batch infill strategy is parametrized by `"q_infill_strategy"` (default: `"KB"`).
The different strategies are:

| Value     | Name                          | Expression                  |
|-----------|-------------------------------|-----------------------------|
| `"CLMIN"` | Minimum constant liar         | $\min \{y_1,\ldots,y_n\}$   |
| `"KB"`    | Kriging believer              | $\mu(x)$                    |
| `"KBLB"`  | Kriging believer lower bound  | $\mu(x)-3\sigma(x)$         |
| `"KBUB"`  | Kriging believer upper bound  | $\mu(x)+3\sigma(x)$         |

### Surrogate models

You can use the option `gp_config` to change the Gaussian process (GP) model
(GP settings, mixture of GPs, type of mixture, hyperparameters optimization, dimension reduction, etc.).

### Initial surrogate model

The initial surrogate is trained from a design of experiments (DOE) of size `n_doe` (default: `10`).
You can also use the option `doe` expecting either

- a DOE of type `ndarray`,
- DOE algorith settings to create the DOE.

### Constraints

For optimization problems including inequality constraints,
the constraint output $g$ are approximated by a GP surrogate model $G$.
Then,
the constraint $g(x)\leq 0$ can be:

- replaced by the mean $\mathbb{E}[G(x)]\leq 0$ (when `cstr_strategy` is `"MC"`),
- replaced by the upper trust bound $\mathbb{E}[G(x)]+2\mathbb{S}[G(x)]\leq 0$ (when `cstr_strategy` is `"UTB"`),
- part of the acquisition criterion,
  as a probability of feasibility $\mathbb{P}[G(x)\leq 0]$ multiplying the acquisition criterion
  related to the objective, e.g., $EI(x)\mathbb{P}[G(x)\leq 0]$ in the case of expected improvement
  (when `cstr_strategy` is `"INFILL"`).

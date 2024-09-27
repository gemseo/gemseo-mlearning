<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->
# SMT's surrogate models

The [surrogate modeling toolbox (SMT)](https://smt.readthedocs.io)
is an open-source Python package for surrogate modeling with a focus on derivatives [@SMT2019,@saves2024smt].

`gemseo-mlearning` proposes the [SMTRegressor][gemseo_mlearning.regression.smt_regressor.SMTRegressor]
to easily use any SMT's surrogate model in your GEMSEO processes.

You only have to instantiate this class
from:

- the [IODataset][gemseo.datasets.io_dataset.IODataset] including your input and output samples,
- the name of a SMT's surrogate model (*e.g.* `"KRG"` for Kriging or `"RBF"` for radial basis function),
- the options of this surrogate model,
- and the usual options of a [BaseRegressor][gemseo.mlearning.regression.algos.base_regressor],
  namely
  `transformer` to transform the input and output data,
  `input_names` to use a subset of input variables and
  `output_names` to use a subset of output variables.

Here's how to build an SMT's RBF surrogate model with the basis function scaling parameter `d_0` set to 2.0 (instead of 1.0):

```python
model = SMTRegressor(training_dataset, "RBF", d0=2.0)
model.learn()
```

Regarding the options,
you will find more information in the [SMT's user guide](https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models.html)
by looking at the tables at the bottom of the pages.

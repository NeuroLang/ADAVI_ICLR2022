# ADAVI: Automatic Dual Amortized Variational Inference

## Install
To install this package run from this directory:
```bash
pip install .
``` 

## Directory organization
* subdirectory `adavi` contains our package
* subdirectory `examples` contains scripts to reproduce experiments (see `README` inside the directory)
* subdirectory `data` contains dataset and can be used to store regenerated experimental data

### adavi.set_transformer

Provides a fully-parametrized Keras implementation of Set Transformers:
> Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks (Lee et al. 2019)

[link to the paper](http://proceedings.mlr.press/v97/lee19d.html)


### adavi.normalizing_flow

Contains tensorflow-probability utilities to construct chains of normalizing flow bijectors

### adavi.dual

Contains our main methodological contribution: the utilities to derive from a `generative_hbm` a _dual_ architecture to perform amortized inference.

Notably provides a TFP - Keras implementation of Cascading Flows:
> Automatic variational inference with cascading flows (Ambrogioni et al. 2021)

[link to the paper](https://arxiv.org/abs/2102.04801)

## Example usage
```python
import tensorflow_probability as tfp
from adavi.dual.models import ADAVFamily

tfd = tfp.distributions
tfb = tfp.bijectors

generative_hbm = tfd.JointDistributionNamed(
    model=dict(
        mu=tfd.Normal(loc=0, scale=1),
        X=lambda mu: tfd.Sample(
            distribution=tfd.Normal(loc=mu, scale=0.1),
            sample_shape=(10,)
        )
    )
)
hbm_kwargs = dict(
    generative_hbm=generative_hbm,
    hierarchies={
        "mu": 1,
        "X": 0
    },
    link_functions={
        "mu": tfb.Identity(),
        "X": tfb.Identity()
    }
)

adav_family = ADAVFamily(
    set_transforer_kwargs={...},
    conditional_nf_chain_kwargs={...},
    **hbm_kwargs
)

train_data = generative_hbm.sample((100,))
val_datum = generative_hbm.sample((1,))

adav_family.compile(
    train_method="reverse_KL",
    n_theta_draws_per_x=32,
    optimizer="adam"
)
adav_family.fit(train_data)
posterior_sample = (
    adav_family
    .sample_parameters_conditioned_to_data(
        val_datum
    )
)
```
# %% Imports
import pickle
from typing import Dict

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

# %% Random seed

# seed = 1234
# tf.random.set_seed(seed)
# np.random.seed(seed)

# %% Generative Hierarchical Bayesian Model

D = 2
G = 10

concentration_a = tf.ones((D,))
rate_a = 0.5

loc_b = tf.ones((D,))
scale_b = 0.3

generative_hbm = tfd.JointDistributionNamed(
    model=dict(
        a=tfd.Independent(
            tfd.Gamma(
                concentration=concentration_a,
                rate=rate_a
            ),
            reinterpreted_batch_ndims=1
        ),
        b=lambda a: tfd.Sample(
            tfd.Independent(
                tfd.Laplace(
                    loc=a,
                    scale=scale_b
                ),
                reinterpreted_batch_ndims=1
            ),
            sample_shape=(G,)
        )
    )
)

link_functions = {
    "a": tfb.Softplus(),
    "b": tfb.Identity()
}

hbm_kwargs = dict(
    generative_hbm=generative_hbm,
    hierarchies={
        "a": 1,
        "b": 0
    },
    link_functions=link_functions
)

total_hbm_kwargs = dict(
    generative_hbm=generative_hbm,
    link_functions=link_functions,
    observed_rv="b"
)

# %% Dataset generation

train_size, val_size = 20_000, 2000
dataset = pickle.load(
    open(
        "../data/NC_dataset.p",
        "rb"
    )
)
train_data = dataset["train"]
val_data = dataset["val"]

# %% Ground HBM, used by CF

ground_hbm = tfd.JointDistributionNamed(
    model=dict(
        a=tfd.Independent(
            tfd.Gamma(
                concentration=concentration_a,
                rate=rate_a
            ),
            reinterpreted_batch_ndims=1
        ),
        **{
            f"b_{g}": lambda a: tfd.Independent(
                tfd.Laplace(
                    loc=a,
                    scale=scale_b
                ),
                reinterpreted_batch_ndims=1
            )
            for g in range(G)
        }
    )
)

cf_hbm_kwargs = dict(
    generative_hbm=ground_hbm,
    observed_rvs=[
        f"b_{g}"
        for g in range(G)
    ],
    link_functions={
        "a": tfb.Softplus(),
        **{
            f"b_{g}": tfb.Identity()
            for g in range(G)
        }
    },
    observed_rv_reshapers={
        f"b_{g}": tfb.Identity()
        for g in range(G)
    }
)

# %% Data reshaping


def stack_data(
    data: Dict[str, tf.Tensor]
) -> Dict[str, tf.Tensor]:
    output_data = {}
    output_data["a"] = data["a"]
    try:
        output_data["b"] = tf.stack(
            [
                data[f"b_{g}"]
                for g in range(G)
            ],
            axis=-2
        )
    except KeyError:
        pass

    return output_data


def slice_data(
    data: Dict[str, tf.Tensor]
) -> Dict[str, tf.Tensor]:
    output_data = {}
    output_data["a"] = data["a"]
    try:
        for g in range(G):
            output_data[f"b_{g}"] = data["b"][..., g, :]
    except KeyError:
        pass

    return output_data


# %% CF Data

cf_train_data = slice_data(train_data)
cf_val_data = slice_data(val_data)

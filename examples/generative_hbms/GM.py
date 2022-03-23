# %% Imports
import pickle
import argparse
from typing import Dict

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

# %% Random seed

# seed = 1234
# tf.random.set_seed(seed)
# np.random.seed(seed)

# %% Argument parsing

parser = argparse.ArgumentParser()
parser.add_argument(
    "--na-val-idx",
    type=int,
    default=4,
    required=False
)
args, _ = parser.parse_known_args()


# %% Generative Hierarchical Bayesian Model

D = 2
L = 3
G = 3
N = 50
loc_mu = tf.zeros((D,))
scale_mu = 1.0
scale_mu_g = 0.2
scale_x = 0.05
dirichlet_concentration = tf.ones((L,)) * 1

generative_hbm = tfd.JointDistributionNamed(
    model=dict(
        mu=tfd.Sample(
            tfd.Independent(
                tfd.Normal(
                    loc=loc_mu,
                    scale=scale_mu
                ),
                reinterpreted_batch_ndims=1
            ),
            sample_shape=(L,),
            name="mu"
        ),
        mu_g=lambda mu: tfd.Sample(
            tfd.Independent(
                tfd.Normal(
                    loc=mu,
                    scale=scale_mu_g
                ),
                reinterpreted_batch_ndims=2
            ),
            sample_shape=(G,),
            name="mu_g"
        ),
        probs=tfd.Sample(
            tfd.Dirichlet(
                concentration=dirichlet_concentration
            ),
            sample_shape=(G,),
            name="probs"
        ),
        x=lambda mu_g, probs: tfd.TransformedDistribution(
            tfd.Sample(
                tfd.Independent(
                    tfd.Mixture(
                        cat=tfd.Categorical(probs=probs),
                        components=[
                            tfd.Independent(
                                tfd.Normal(
                                    loc=mu_g[..., i, :],
                                    scale=scale_x
                                ),
                                reinterpreted_batch_ndims=1
                            )
                            for i in range(L)
                        ]
                    ),
                    reinterpreted_batch_ndims=1
                ),
                sample_shape=(N,)
            ),
            bijector=tfb.Transpose(perm=[1, 0, 2]),
            name="x"
        )
    )
)

link_functions = {
    "x": tfb.Identity(),
    "mu_g": tfb.Identity(),
    "probs": tfb.SoftmaxCentered(),
    "mu": tfb.Identity()
}

hbm_kwargs = dict(
    generative_hbm=generative_hbm,
    hierarchies={
        "x": 0,
        "mu_g": 1,
        "probs": 1,
        "mu": 2
    },
    link_functions=link_functions
)

total_hbm_kwargs = dict(
    generative_hbm=generative_hbm,
    link_functions=link_functions,
    observed_rv="x"
)

faithful_hbm_kwargs = dict(
    generative_hbm=generative_hbm,
    observed_rvs=['x'],
    plate_cardinalities={
        'G': G,
        'N': N
    },
    link_functions=link_functions,
    observed_rv_reshapers={
        "x": tfb.Identity()
    },
    plates_per_rv={
        "mu": tuple(),
        "mu_g": ('G',),
        "probs": ('G',),
        "x": ('G', 'N',)
    },
)

# %% Dataset generation

dataset = pickle.load(
    open("../data/GM_dataset.p", "rb")
)
train_data = dataset["train"]
val_data = dataset["val"]

# %% Prior, used by SBI

na_val_idx = args.na_val_idx

generative_prior = tfd.JointDistributionNamed(
    model=dict(
        mu=tfd.Sample(
            tfd.Independent(
                tfd.Normal(
                    loc=loc_mu,
                    scale=scale_mu
                ),
                reinterpreted_batch_ndims=1
            ),
            sample_shape=(L,),
            name="mu"
        ),
        mu_g=lambda mu: tfd.Sample(
            tfd.Independent(
                tfd.Normal(
                    loc=mu,
                    scale=scale_mu_g
                ),
                reinterpreted_batch_ndims=2
            ),
            sample_shape=(G,),
            name="mu_g"
        ),
        probs=tfd.Sample(
            tfd.Dirichlet(
                concentration=dirichlet_concentration
            ),
            sample_shape=(G,),
            name="probs"
        )
    )
)

# %% Ground graph, used by CF

ground_hbm = tfd.JointDistributionNamed(
    model={
        "mu": tfd.Sample(
            tfd.Independent(
                tfd.Normal(
                    loc=loc_mu,
                    scale=scale_mu
                ),
                reinterpreted_batch_ndims=1
            ),
            sample_shape=(L,)
        ),
        **{
            f"mu_{g}": lambda mu: tfd.Independent(
                tfd.Normal(
                    loc=mu,
                    scale=scale_mu_g
                ),
                reinterpreted_batch_ndims=2
            )
            for g in range(G)
        },
        **{
            f"probs_{g}": tfd.Dirichlet(
                concentration=dirichlet_concentration
            )
            for g in range(G)
        },
        **{
            f"mu_{g}_{n}": eval(
                f"""lambda mu_{g}, probs_{g}: tfd.Mixture(
                    cat=tfd.Categorical(probs=probs_{g}),
                    components=[
                        tfd.Independent(
                            tfd.Normal(
                                loc=mu_{g}[..., i, :],
                                scale=scale_x
                            ),
                            reinterpreted_batch_ndims=1
                        )
                        for i in range(L)
                    ]
                    )"""
            )
            for g in range(G)
            for n in range(N)
        }
    }
)

cf_hbm_kwargs = dict(
    generative_hbm=ground_hbm,
    observed_rvs=[
        f"mu_{g}_{n}"
        for g in range(G)
        for n in range(N)
    ],
    link_functions={
        "mu": tfb.Identity(),
        **{
            f"mu_{g}": tfb.Identity()
            for g in range(G)
        },
        **{
            f"probs_{g}": tfb.SoftmaxCentered()
            for g in range(G)
        },
        **{
            f"mu_{g}_{n}": tfb.Identity()
            for g in range(G)
            for n in range(N)
        }
    },
    observed_rv_reshapers={
        f"mu_{g}_{n}": tfb.Identity()
        for g in range(G)
        for n in range(N)
    }
)

# %% Data reshaping


def stack_data(
    data: Dict[str, tf.Tensor]
) -> Dict[str, tf.Tensor]:
    output_data = {}
    output_data["mu"] = data["mu"]
    output_data["mu_g"] = tf.stack(
        [
            data[f"mu_{g}"]
            for g in range(G)
        ],
        axis=-3
    )
    output_data["probs"] = tf.stack(
        [
            data[f"probs_{g}"]
            for g in range(G)
        ],
        axis=-2
    )
    try:
        output_data["x"] = tf.stack(
            [
                tf.stack(
                    [
                        data[f"mu_{g}_{n}"]
                        for n in range(N)
                    ],
                    axis=-2
                )
                for g in range(G)
            ],
            axis=-3
        )
    except KeyError:
        pass

    return output_data


def slice_data(
    data: Dict[str, tf.Tensor]
) -> Dict[str, tf.Tensor]:
    output_data = {}
    output_data["mu"] = data["mu"]
    for g in range(G):
        output_data[f"mu_{g}"] = data["mu_g"][..., g, :, :]
        output_data[f"probs_{g}"] = data["probs"][..., g, :]
    try:
        for g in range(G):
            for n in range(N):
                output_data[f"mu_{g}_{n}"] = data["x"][..., g, n, :]
    except KeyError:
        pass

    return output_data


# %% CF Data

cf_train_data = slice_data(train_data)
cf_val_data = slice_data(val_data)

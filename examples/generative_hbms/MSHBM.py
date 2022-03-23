# %% Imports
from typing import Iterable, Dict

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors
Root = tfp.distributions.JointDistributionCoroutine.Root

# %% Random seed

seed = 1234
tf.random.set_seed(seed)
np.random.seed(seed)

# %% Positive qUadrant unit sphere link funtion

unit_sphere_normalizer = (
    tfd.VonMisesFisher(
        mean_direction=tf.constant([1, 0]),
        concentration=1
    )
    .experimental_default_event_space_bijector()
)

# %% Generative Hierarchical Bayesian Model

N, T, S, D, L = 5, 2, 2, 2, 2

mu_g_low = -4
mu_g_high = 4

kappa_low = -4
kappa_high = -3
sigma_low = -3
sigma_high = -2
epsilon_low = -2
epsilon_high = -1

concentration = 2 * tf.ones((L,))


def repeat_to_shape(
    x: tf.Tensor,
    target_shape: Iterable,
    axis: int
) -> tf.Tensor:
    out = x
    for size in target_shape:
        out = tf.repeat(
            tf.expand_dims(
                out,
                axis=axis
            ),
            (size,),
            axis=axis
        )

    return out


generative_hbm = tfd.JointDistributionNamed(
    model=dict(
        mu_g=tfd.Sample(
            tfd.TransformedDistribution(
                tfd.Independent(
                    tfd.Uniform(
                        low=mu_g_low * tf.ones((D - 1,)),
                        high=mu_g_high * tf.ones((D - 1,)),
                    ),
                    reinterpreted_batch_ndims=1
                ),
                bijector=unit_sphere_normalizer
            ),
            sample_shape=(L,),
            name="mu_g"
        ),
        epsilon=tfd.TransformedDistribution(
            tfd.Sample(
                tfd.Uniform(
                    low=epsilon_low,
                    high=epsilon_high
                ),
                sample_shape=(L,)
            ),
            bijector=tfb.Exp(),
            name="epsilon"
        ),
        mu_s=lambda mu_g, epsilon: tfd.Sample(
            tfd.TransformedDistribution(
                tfd.Independent(
                    tfd.Normal(
                        loc=(
                            unit_sphere_normalizer
                            .inverse(mu_g)
                        ),
                        scale=repeat_to_shape(
                            epsilon,
                            target_shape=(1,),
                            axis=-1
                        )
                    ),
                    reinterpreted_batch_ndims=2
                ),
                bijector=unit_sphere_normalizer
            ),
            sample_shape=(S,),
            name="mu_s"
        ),
        sigma=tfd.TransformedDistribution(
            tfd.Sample(
                tfd.Uniform(
                    low=sigma_low,
                    high=sigma_high
                ),
                sample_shape=(L,)
            ),
            bijector=tfb.Exp(),
            name="sigma"
        ),
        mu_s_t=lambda mu_s, sigma: tfd.TransformedDistribution(
            tfd.Sample(
                tfd.TransformedDistribution(
                    tfd.Independent(
                        tfd.Normal(
                            loc=(
                                unit_sphere_normalizer
                                .inverse(mu_s)
                            ),
                            scale=repeat_to_shape(
                                repeat_to_shape(
                                    sigma,
                                    target_shape=(S,),
                                    axis=-2
                                ),
                                target_shape=(1,),
                                axis=-1
                            )
                        ),
                        reinterpreted_batch_ndims=3
                    ),
                    bijector=unit_sphere_normalizer
                ),
                sample_shape=(T,)
            ),
            bijector=tfb.Transpose([1, 0, 2, 3]),
            name="mu_s_t"
        ),
        kappa=tfd.TransformedDistribution(
            tfd.Uniform(
                low=sigma_low,
                high=sigma_high
            ),
            bijector=tfb.Exp(),
            name="kappa"
        ),
        probs=tfd.Dirichlet(
            concentration=concentration,
            name="probs"
        ),
        X_s_t=lambda mu_s_t, kappa, probs: tfd.TransformedDistribution(
            tfd.Sample(
                tfd.Mixture(
                    cat=tfd.Categorical(
                        probs=probs
                    ),
                    components=[
                        tfd.Independent(
                            tfd.Normal(
                                loc=(
                                    unit_sphere_normalizer
                                    .inverse(mu_s_t)
                                    [..., l, :]
                                ),
                                scale=repeat_to_shape(
                                    kappa,
                                    target_shape=(S, T, 1),
                                    axis=-1
                                )
                            ),
                            reinterpreted_batch_ndims=3
                        )
                        for l in range(L)
                    ]
                ),
                sample_shape=(N,)
            ),
            bijector=tfb.Chain(
                bijectors=[
                    unit_sphere_normalizer,
                    tfb.Transpose([1, 2, 0, 3])
                ]
            ),
            name="X_s_t"
        )
    )
)

hbm_kwargs = dict(
    generative_hbm=generative_hbm,
    hierarchies={
        "mu_g": 3,
        "epsilon": 3,
        "mu_s": 2,
        "sigma": 3,
        "mu_s_t": 1,
        "kappa": 3,
        "probs": 3,
        "X_s_t": 0
    },
    link_functions={
        "mu_g": tfb.Chain(
            [
                unit_sphere_normalizer,
                tfb.SoftClip(
                    low=mu_g_low,
                    high=mu_g_high
                )
            ]
        ),
        "epsilon": tfb.Chain(
            [
                tfb.Exp(),
                tfb.SoftClip(
                    low=epsilon_low,
                    high=epsilon_high
                )
            ]
        ),
        "mu_s": unit_sphere_normalizer,
        "sigma": tfb.Chain(
            [
                tfb.Exp(),
                tfb.SoftClip(
                    low=sigma_low,
                    high=sigma_high
                )
            ]
        ),
        "mu_s_t": unit_sphere_normalizer,
        "kappa": tfb.Chain(
            [
                tfb.Exp(),
                tfb.SoftClip(
                    low=kappa_low,
                    high=kappa_high
                )
            ]
        ),
        "probs": tfb.SoftmaxCentered(),
        "X_s_t": unit_sphere_normalizer
    }
)

faithful_hbm_kwargs = dict(
    generative_hbm=generative_hbm,
    observed_rvs=["X_s_t"],
    plates=["S", "T", "N"],
    plates_per_rv={
        "mu_g": [],
        "epsilon": [],
        "mu_s": ["S"],
        "sigma": [],
        "mu_s_t": ["S", "T"],
        "kappa": [],
        "probs": [],
        "X_s_t": ["S", "T", "N"]
    },
    link_functions={
        "mu_g": tfb.Chain(
            [
                unit_sphere_normalizer,
                tfb.SoftClip(
                    low=mu_g_low,
                    high=mu_g_high
                )
            ]
        ),
        "epsilon": tfb.Chain(
            [
                tfb.Exp(),
                tfb.SoftClip(
                    low=epsilon_low,
                    high=epsilon_high
                )
            ]
        ),
        "mu_s": unit_sphere_normalizer,
        "sigma": tfb.Chain(
            [
                tfb.Exp(),
                tfb.SoftClip(
                    low=sigma_low,
                    high=sigma_high
                )
            ]
        ),
        "mu_s_t": unit_sphere_normalizer,
        "kappa": tfb.Chain(
            [
                tfb.Exp(),
                tfb.SoftClip(
                    low=kappa_low,
                    high=kappa_high
                )
            ]
        ),
        "probs": tfb.SoftmaxCentered(),
        "X_s_t": unit_sphere_normalizer
    },
    observed_rv_reshapers={
        "X_s_t": tfb.Identity()
    },
)

# # %% Dataset generation

train_size, val_size = 20_000, 2000
train_data, val_data = (
    generative_hbm.sample(size)
    for size in [train_size, val_size]
)

# %% Ground graph, used by CF

ground_hbm = tfd.JointDistributionNamed(
    model={
        "mu_g": tfd.Sample(
            tfd.TransformedDistribution(
                tfd.Independent(
                    tfd.Uniform(
                        low=mu_g_low * tf.ones((D - 1,)),
                        high=mu_g_high * tf.ones((D - 1,)),
                    ),
                    reinterpreted_batch_ndims=1
                ),
                bijector=unit_sphere_normalizer
            ),
            sample_shape=(L,),
            name="mu_g"
        ),
        "epsilon": tfd.TransformedDistribution(
            tfd.Sample(
                tfd.Uniform(
                    low=epsilon_low,
                    high=epsilon_high
                ),
                sample_shape=(L,)
            ),
            bijector=tfb.Exp(),
            name="epsilon"
        ),
        **{
            f"mu_{s}": lambda mu_g, epsilon: tfd.TransformedDistribution(
                tfd.Independent(
                    tfd.Normal(
                        loc=(
                            unit_sphere_normalizer
                            .inverse(mu_g)
                        ),
                        scale=tf.expand_dims(
                            epsilon,
                            axis=-1
                        )
                    ),
                    reinterpreted_batch_ndims=2
                ),
                bijector=unit_sphere_normalizer
            )
            for s in range(S)
        },
        "sigma": tfd.TransformedDistribution(
            tfd.Sample(
                tfd.Uniform(
                    low=sigma_low,
                    high=sigma_high
                ),
                sample_shape=(L,)
            ),
            bijector=tfb.Exp(),
            name="sigma"
        ),
        **{
            f"mu_{s}_{t}": eval(
                f"""lambda mu_{s}, sigma: tfd.TransformedDistribution(
                    tfd.Independent(
                        tfd.Normal(
                            loc=(
                                unit_sphere_normalizer
                                .inverse(mu_{s})
                            ),
                            scale=tf.expand_dims(
                                sigma,
                                axis=-1
                            )
                        ),
                        reinterpreted_batch_ndims=2
                    ),
                    bijector=unit_sphere_normalizer
                )"""
            )
            for s in range(S)
            for t in range(T)
        },
        "kappa": tfd.TransformedDistribution(
            tfd.Uniform(
                low=sigma_low,
                high=sigma_high
            ),
            bijector=tfb.Exp(),
            name="kappa"
        ),
        "probs": tfd.Dirichlet(
            concentration=concentration,
            name="probs"
        ),
        **{
            f"X_{s}_{t}_{n}": eval(
                f"""lambda mu_{s}_{t}, kappa, probs: tfd.TransformedDistribution(
                    tfd.Mixture(
                        cat=tfd.Categorical(
                            probs=probs
                        ),
                        components=[
                            tfd.Independent(
                                tfd.Normal(
                                    loc=(
                                        unit_sphere_normalizer
                                        .inverse(mu_{s}_{t})
                                        [..., l, :]
                                    ),
                                    scale=tf.expand_dims(
                                        kappa,
                                        axis=-1
                                    )
                                ),
                                reinterpreted_batch_ndims=1
                            )
                            for l in range(L)
                        ]
                    ),
                    bijector=unit_sphere_normalizer
                )"""
            )
            for s in range(S)
            for t in range(T)
            for n in range(N)
        }
    }
)

cf_hbm_kwargs = dict(
    generative_hbm=ground_hbm,
    observed_rvs=[
        f"X_{s}_{t}_{n}"
        for s in range(S)
        for t in range(T)
        for n in range(N)
    ],
    link_functions={
        "mu_g": tfb.Chain(
            [
                unit_sphere_normalizer,
                tfb.SoftClip(
                    low=mu_g_low,
                    high=mu_g_high
                )
            ]
        ),
        "epsilon": tfb.Chain(
            [
                tfb.Exp(),
                tfb.SoftClip(
                    low=epsilon_low,
                    high=epsilon_high
                )
            ]
        ),
        **{
            f"mu_{s}": unit_sphere_normalizer
            for s in range(S)
        },
        "sigma": tfb.Chain(
            [
                tfb.Exp(),
                tfb.SoftClip(
                    low=sigma_low,
                    high=sigma_high
                )
            ]
        ),
        **{
            f"mu_{s}_{t}": unit_sphere_normalizer
            for s in range(S)
            for t in range(T)
        },
        "kappa": tfb.Chain(
            [
                tfb.Exp(),
                tfb.SoftClip(
                    low=kappa_low,
                    high=kappa_high
                )
            ]
        ),
        "probs": tfb.SoftmaxCentered(),
        **{
            f"X_{s}_{t}_{n}": unit_sphere_normalizer
            for s in range(S)
            for t in range(T)
            for n in range(N)
        },
    },
    observed_rv_reshapers={
        f"X_{s}_{t}_{n}": tfb.Identity()
        for s in range(S)
        for t in range(T)
        for n in range(N)
    }
)

# %% Data reshaping


def stack_data(
    data: Dict[str, tf.Tensor]
) -> Dict[str, tf.Tensor]:
    output_data = {}
    output_data["mu_g"] = data["mu_g"]
    output_data["epsilon"] = data["epsilon"]
    output_data["mu_s"] = tf.stack(
        [
            data[f"mu_{s}"]
            for s in range(S)
        ],
        axis=-3
    )
    output_data["sigma"] = data["sigma"]
    output_data["mu_s_t"] = tf.stack(
        [
            tf.stack(
                [
                    data[f"mu_{s}_{t}"]
                    for t in range(T)
                ],
                axis=-3
            )
            for s in range(S)
        ],
        axis=-4
    )
    output_data["kappa"] = data["kappa"]
    output_data["probs"] = data["probs"]
    try:
        output_data["X_s_t"] = tf.stack(
            [
                tf.stack(
                    [
                        tf.stack(
                            [
                                data[f"X_{s}_{t}_{n}"]
                                for n in range(N)
                            ],
                            axis=-2
                        )
                        for t in range(T)
                    ],
                    axis=-3
                )
                for s in range(S)
            ],
            axis=-4
        )
    except KeyError:
        pass

    return output_data


def slice_data(
    data: Dict[str, tf.Tensor]
) -> Dict[str, tf.Tensor]:
    output_data = {}

    output_data["mu_g"] = data["mu_g"]
    output_data["epsilon"] = data["epsilon"]
    output_data["sigma"] = data["sigma"]
    output_data["kappa"] = data["kappa"]
    output_data["probs"] = data["probs"]

    for s in range(S):
        output_data[f"mu_{s}"] = data["mu_s"][..., s, :, :]
        for t in range(T):
            output_data[f"mu_{s}_{t}"] = data["mu_s_t"][..., s, t, :, :]
            try:
                for n in range(N):
                    output_data[f"X_{s}_{t}_{n}"] = (
                        data["X_s_t"][..., s, t, n, :]
                    )
            except KeyError:
                pass

    return output_data


# %% CF Data

cf_train_data = slice_data(train_data)
cf_val_data = slice_data(val_data)
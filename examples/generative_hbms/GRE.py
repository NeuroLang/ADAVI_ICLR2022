# %% Imports
from typing import Dict
import pickle
import argparse

import tensorflow as tf
import tensorflow_probability as tfp

from adavi.dual.models import ADAVFamily

tfd = tfp.distributions
tfb = tfp.bijectors

# %% Random seed

# seed = 1234
# tf.random.set_seed(seed)
# np.random.seed(seed)

# %% Argument parsing

parser = argparse.ArgumentParser()
parser.add_argument(
    "--G",
    type=int,
    default=3,
    required=False
)
parser.add_argument(
    "--na-val-idx",
    type=int,
    default=4,
    required=False
)
args, _ = parser.parse_known_args()

# %% Generative Hierarchical Bayesian Model

D = 2
G = args.G
N = 50
loc_mu = tf.zeros((D,))
scale_mu = 0.5
scale_mu_g = 0.5
scale_x = 0.1

generative_hbm = tfd.JointDistributionNamed(
    model=dict(
        mu=tfd.Independent(
            tfd.Normal(
                loc=loc_mu,
                scale=scale_mu
            ),
            reinterpreted_batch_ndims=1,
            name="mu"
        ),
        mu_g=lambda mu: tfd.Sample(
            tfd.Independent(
                tfd.Normal(
                    loc=mu,
                    scale=scale_mu_g
                ),
                reinterpreted_batch_ndims=1
            ),
            sample_shape=(G,),
            name="mu_g"
        ),
        x=lambda mu_g: tfd.TransformedDistribution(
            tfd.Sample(
                tfd.Independent(
                    tfd.Normal(
                        loc=mu_g,
                        scale=scale_x
                    ),
                    reinterpreted_batch_ndims=2
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
    "mu": tfb.Identity()
}

hbm_kwargs = dict(
    generative_hbm=generative_hbm,
    hierarchies={
        "x": 0,
        "mu_g": 1,
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
        "x": ('G', 'N')
    },
)

# %% Dataset generation

try:
    dataset = pickle.load(
        open(f"../data/GRE_dataset_G{G}.p", "rb")
    )
    train_data = dataset["train"]
    val_data = dataset["val"]
except FileNotFoundError:
    seed = 1234
    tf.random.set_seed(seed)
    train_size, val_size = 20_000, 2000
    train_data, val_data = (
        generative_hbm.sample(size)
        for size in [train_size, val_size]
    )
    dataset = {
        data_key: {
            key: value.numpy()
            for key, value in data.items()
        }
        for data_key, data in [
            ("train", train_data),
            ("val", val_data)
        ]
    }
    pickle.dump(
        dataset,
        open(f"../data/GRE_dataset_G{G}.p", "wb")
    )

# %%


def get_mean_KL_divergence(
    adav_family: ADAVFamily,
    data: Dict
) -> tf.Tensor:
    """Calculates analytical KL divergence
    and averages it over a validation
    dataset
    # ! Assumes a single bijector in adav_family
    # ! to be a conditional affine with tril scale

    Parameters
    ----------
    adav_family : ADAVFamily
        the architecture to validate
    data : Dict
        data over which to average KL divergence

    Returns
    -------
    tf.Tensor
        mean KL divergence
    """
    mean_KL_divergence = 0
    group_means = tf.reduce_mean(
        data["x"],
        axis=-2
    )
    encodings = adav_family.encode_data(
        x=data["x"]
    )
    variational_posterior_mu_g = tfd.MultivariateNormalTriL(
        loc=(
            adav_family
            .conditional_density_estimators["mu_g"]
            .bijector
            .bijectors[-1]
            .bijectors[-1]
            .shift(
                encodings[1]
            )
        ),
        scale_tril=tfp.math.fill_triangular(
            adav_family
            .conditional_density_estimators["mu_g"]
            .bijector
            .bijectors[-1]
            .bijectors[-1]
            .scale(
                encodings[1]
            )
        ),
    )
    analytical_posterior_mu_g = tfd.MultivariateNormalDiag(
        loc=group_means,
        scale_diag=[scale_x / N**0.5] * D
    )
    mean_KL_divergence += tf.reduce_mean(
        tfd.kl_divergence(
            variational_posterior_mu_g,
            analytical_posterior_mu_g
        )
    )

    variational_posterior_mu = tfd.MultivariateNormalTriL(
        loc=(
            adav_family
            .conditional_density_estimators["mu"]
            .bijector
            .bijectors[-1]
            .bijectors[-1]
            .shift(
                encodings[2]
            )
        ),
        scale_tril=tfp.math.fill_triangular(
            adav_family
            .conditional_density_estimators["mu"]
            .bijector
            .bijectors[-1]
            .bijectors[-1]
            .scale(
                encodings[2]
            )
        ),
    )

    population_mean = tf.reduce_mean(
        group_means,
        axis=-2
    )
    posterior_mean = population_mean / (1 + scale_mu_g**2/(G * scale_mu**2))
    posterior_scale = (1/(1/scale_mu**2 + G/scale_mu_g**2))**0.5

    analytical_posterior_mu = tfd.MultivariateNormalDiag(
        loc=posterior_mean,
        scale_diag=[posterior_scale] * D
    )

    mean_KL_divergence += tf.reduce_mean(
        tfd.kl_divergence(
            variational_posterior_mu,
            analytical_posterior_mu
        )
    )

    return mean_KL_divergence


# %% Prior, used by SBI

na_val_idx = args.na_val_idx

generative_prior = tfd.JointDistributionNamed(
    model=dict(
        mu=tfd.Independent(
            tfd.Normal(
                loc=loc_mu,
                scale=scale_mu
            ),
            reinterpreted_batch_ndims=1,
            name="mu"
        ),
        mu_g=lambda mu: tfd.Sample(
            tfd.Independent(
                tfd.Normal(
                    loc=mu,
                    scale=scale_mu_g
                ),
                reinterpreted_batch_ndims=1
            ),
            sample_shape=(G,),
            name="mu_g"
        )
    )
)

# %% Ground graph, used by CF

ground_hbm = tfd.JointDistributionNamed(
    model={
        "mu": tfd.Independent(
            tfd.Normal(
                loc=loc_mu,
                scale=scale_mu
            ),
            reinterpreted_batch_ndims=1
        ),
        **{
            f"mu_{g}": lambda mu: tfd.Independent(
                tfd.Normal(
                    loc=mu,
                    scale=scale_mu_g
                ),
                reinterpreted_batch_ndims=1
            )
            for g in range(G)
        },
        **{
            f"mu_{g}_{n}": eval(
                f"""lambda mu_{g}: tfd.Independent(
                    tfd.Normal(
                        loc=mu_{g},
                        scale=scale_x
                    ),
                    reinterpreted_batch_ndims=1
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
        output_data[f"mu_{g}"] = data["mu_g"][..., g, :]
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

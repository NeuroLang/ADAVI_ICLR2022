# %% Imports
from collections import defaultdict
import pickle
import os
from time import process_time

import tensorflow as tf


import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from adavi.dual.models import (
    ADAVFamily
)
from generative_hbms.GRE import (
    D, G, N, scale_mu, scale_mu_g, scale_x,
    generative_hbm,
    train_data,
    val_data,
)

tfd = tfp.distributions
tfb = tfp.bijectors


# %% Variables

samples = {}
n_val_examples = 20

# %% Mean Field VI baseline

samples = {}
times = []
losses = []
for val_idx in range(0, n_val_examples):

    def log_prob_fn(mu, mu_g):
        return generative_hbm.log_prob(
            mu=mu,
            mu_g=mu_g,
            x=val_data["x"][val_idx]
        )

    q = tfd.JointDistributionNamed(
        model=dict(
            mu=tfd.Independent(
                tfd.Normal(
                    loc=tf.Variable(
                        initial_value=tf.zeros((D,)),
                        name="q_mu_loc"
                    ),
                    scale=tfp.util.TransformedVariable(
                        initial_value=1.,
                        bijector=tfb.Softplus(),
                        name='q_mu_scale'
                    ),
                ),
                reinterpreted_batch_ndims=1,
                name="mu"
            ),
            mu_g=tfd.Independent(
                tfd.Normal(
                    loc=tf.Variable(
                        initial_value=tf.zeros((G, D)),
                        name="q_mu_loc"
                    ),
                    scale=tfp.util.TransformedVariable(
                        initial_value=1.,
                        bijector=tfb.Softplus(),
                        name='q_mu_scale'
                    ),
                ),
                reinterpreted_batch_ndims=2,
                name="mu_g"
            ),
        )
    )

    optimizer = tf.optimizers.Adam(learning_rate=1e-2)

    @tf.function
    def fit_vi():
        return tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn=log_prob_fn,
            surrogate_posterior=q,
            optimizer=optimizer,
            num_steps=10_000,
            sample_size=32
        )
    start_time = process_time()
    mean_field_loss = fit_vi()
    stop_time = process_time()

    samples[val_idx] = {
        key: value.numpy()
        for key, value in q.sample(1000).items()
    }
    print(f"""
        Idx:  {val_idx}
        Time: {stop_time - start_time}
        Loss: {mean_field_loss[-1]}
    """)

    times.append(stop_time - start_time)
    losses.append(mean_field_loss[-1])

# %% Plotting

fig, axs = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(20, 30)
)
for val_idx in range(0, 3):
    group_means = []
    circles = []
    for g in range(G):
        axs[val_idx, 0].scatter(
            val_data["x"][val_idx, g, :, 0],
            val_data["x"][val_idx, g, :, 1],
            color=f"C{g}",
            alpha=0.5
        )

        mean = tf.reduce_mean(
            val_data["x"][val_idx, g],
            axis=-2
        )

        circles.append(
            plt.Circle(
                (mean[0], mean[1]),
                2 * scale_x / N**0.5,
                fill=False,
                color="black",
            )
        )

        group_means.append(mean)

    population_mean = tf.reduce_mean(
        tf.stack(
            group_means,
            axis=-2
        ),
        axis=-2
    )
    posterior_mean = population_mean / (1 + scale_mu_g**2/(G * scale_mu**2))
    posterior_scale = (1/(1/scale_mu**2 + G/scale_mu_g**2))**0.5

    circle = plt.Circle(
        (posterior_mean[0], posterior_mean[1]),
        2 * posterior_scale,
        fill=False,
        color="black"
    )

    axs[val_idx, 0].axis("equal")
    axs[val_idx, 0].set_ylabel(
        f"Example {val_idx}",
        fontsize=30,
        rotation=0,
        ha="right",
        va="center"
    )
    plt.draw()
    x_lim = axs[val_idx, 0].get_xlim()
    y_lim = axs[val_idx, 0].get_ylim()

    axs[val_idx, 1].scatter(
        samples[val_idx]["mu"][:, 0],
        samples[val_idx]["mu"][:, 1],
        color="black",
        s=20,
        alpha=0.5
    )
    axs[val_idx, 1].add_patch(circle)

    for g in range(G):
        axs[val_idx, 1].scatter(
            samples[val_idx]["mu_g"][:, g, 0],
            samples[val_idx]["mu_g"][:, g, 1],
            color=f"C{g}",
            s=20,
            alpha=0.5
        )
        axs[val_idx, 1].add_patch(circles[g])

    axs[val_idx, 1].set_xlim(x_lim)
    axs[val_idx, 1].set_ylim(y_lim)
    axs[val_idx, 1].tick_params(
        which="major",
        labelsize=30
    )

axs[0, 0].set_title(
    "Data",
    fontsize=30
)
axs[0, 1].set_title(
    "reverse KL",
    fontsize=30
)
plt.show()

# %% Storing

base_name = "../data/GM_MF-VI_"
pickle.dump(
   samples,
   open(
        base_name+"samples.p",
        "wb"
    )
)

print(
    f"MF VI G{G} N{N} "
    f"mean time: {tf.reduce_mean(times)}, "
    f"std time: {tf.math.reduce_std(times)}"

    f"mean loss: {tf.reduce_mean(losses)}, "
    f"std loss: {tf.math.reduce_std(losses)}"
)

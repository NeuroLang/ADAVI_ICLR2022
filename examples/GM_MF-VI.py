# %% Imports
import pickle
import os
from time import process_time

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from generative_hbms.GM import (
    L, D, G,
    generative_hbm,
    val_data,
)

tfd = tfp.distributions
tfb = tfp.bijectors


# %% Variables

samples = {}

# %% Mean Field VI baseline

samples = {}
times = []
losses = []
for val_idx in range(0, 20):

    def log_prob_fn(mu, mu_g, probs):
        return generative_hbm.log_prob(
            mu=mu,
            mu_g=mu_g,
            probs=probs,
            x=val_data["x"][val_idx]
        )

    q = tfd.JointDistributionNamed(
        model=dict(
            mu=tfd.Independent(
                tfd.Normal(
                    loc=tf.Variable(
                        initial_value=tf.zeros((L, D,)),
                        name="q_mu_loc"
                    ),
                    scale=tfp.util.TransformedVariable(
                        initial_value=1.,
                        bijector=tfb.Softplus(),
                        name='q_mu_scale'
                    ),
                ),
                reinterpreted_batch_ndims=2,
                name="mu"
            ),
            mu_g=tfd.Independent(
                tfd.Normal(
                    loc=tf.Variable(
                        initial_value=tf.zeros((G, L, D)),
                        name="q_mu_g_loc"
                    ),
                    scale=tfp.util.TransformedVariable(
                        initial_value=1.,
                        bijector=tfb.Softplus(),
                        name='q_mu_g_scale'
                    ),
                ),
                reinterpreted_batch_ndims=3,
                name="mu_g"
            ),
            probs=tfd.Independent(
                tfd.Dirichlet(
                    concentration=tfp.util.TransformedVariable(
                        initial_value=tf.ones((G, L)),
                        bijector=tfb.Softplus(),
                        name="q_concentration"
                    )
                ),
                reinterpreted_batch_ndims=1,
                name="probs"
            )
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

# %%

val_idxs = [0, 1, 2, 3, 4]

fig, axs = plt.subplots(
    nrows=len(val_idxs),
    ncols=2,
    figsize=(20, 10 * len(val_idxs))
)
for idx, val_idx in enumerate(val_idxs):
    for g in range(G):
        axs[idx, 0].scatter(
            val_data["x"][val_idx, g, :, 0],
            val_data["x"][val_idx, g, :, 1],
            color=f"C{g}",
            alpha=0.5
        )

    axs[idx, 0].axis("equal")
    axs[idx, 0].set_ylabel(
        f"{val_idx}",
        fontsize=30,
        rotation=0
    )
    plt.draw()
    x_lim = axs[idx, 0].get_xlim()
    y_lim = axs[idx, 0].get_ylim()

    axs[idx, 1].scatter(
        samples[val_idx]["mu"][:, :, 0],
        samples[val_idx]["mu"][:, :, 1],
        color="black",
        s=20,
        alpha=0.05
    )
    for g in range(G):
        axs[idx, 1].scatter(
            samples[val_idx]["mu_g"][:, g, :, 0],
            samples[val_idx]["mu_g"][:, g, :, 1],
            color=f"C{g}",
            s=20,
            alpha=0.05
        )
    axs[idx, 1].set_xlim(x_lim)
    axs[idx, 1].set_ylim(y_lim)

axs[0, 0].set_title(
    "Data",
    fontsize=30
)
axs[0, 1].set_title(
    "Posterior samples",
    fontsize=30
)
plt.show()

# %%

base_name = "../data/GM_MF-VI_"
pickle.dump(
   samples,
   open(
        base_name+"samples.p",
        "wb"
    )
)

print(
    f"MF VI G{G}"
    f"mean time: {tf.reduce_mean(times)}, "
    f"std time: {tf.math.reduce_std(times)}"

    f"mean loss: {tf.reduce_mean(losses)}, "
    f"std loss: {tf.math.reduce_std(losses)}"
)

# %%

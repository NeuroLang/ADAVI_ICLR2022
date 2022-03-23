# %% Imports
import pickle
from time import time

import tensorflow as tf
import matplotlib.pyplot as plt

from adavi.dual.models import (
    CascadingFlows
)
from generative_hbms.GM import (
    G, N, scale_mu, scale_mu_g, scale_x,
    cf_hbm_kwargs,
    cf_train_data,
    cf_val_data,
    val_data,
    stack_data
)


# %% CascadingFlows kwargs

d = 8

rff_kwargs = dict(
    units_per_layers=[d]
)

cf_kwargs = dict(
    **cf_hbm_kwargs,
    auxiliary_variables_size=d,
    rff_kwargs=rff_kwargs,
    nf_kwargs={},
    amortized=True,
    auxiliary_target_type="identity"
)

# %% CF build
cf = CascadingFlows(
    **cf_kwargs,
)
cf.compile(
    train_method="reverse_KL",
    n_theta_draws_per_x=32,
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2)
)

# %% We fit the training data - amortized
time_1 = time()
history = cf.fit(
    cf_train_data,
    batch_size=32,
    epochs=200,
    shuffle=True,
    verbose=2
)

# %%

time_2 = time()
print(f"Training: {time_2 - time_1}")

# %%  Sampling

samples = {}
losses = []
n_draws = 1_000

for val_idx in range(20):
    repeated_observed_data = {
        f"mu_{g}_{n}": tf.repeat(
            cf_val_data[f"mu_{g}_{n}"][val_idx:val_idx + 1],
            repeats=(n_draws,),
            axis=0
        )
        for g in range(G)
        for n in range(N)
    }
    (
        parameters_sample,
        augmented_posterior_values,
        _,
        auxiliary_values
    ) = cf.sample_parameters_conditioned_to_data(
        data=repeated_observed_data,
        return_internals=True
    )
    samples[val_idx] = stack_data(parameters_sample)

    p = cf.generative_hbm.log_prob(
        **parameters_sample,
        **{
            observed_rv: repeated_observed_data[observed_rv]
            for observed_rv in cf.observed_rvs
        }
    )
    r = cf.MF_log_prob(
        augmented_posterior_values=augmented_posterior_values,
        auxiliary_values=auxiliary_values,
    )
    q = (
        cf
        .joint_log_prob_conditioned_to_data(
            data={
                **parameters_sample,
                **{
                    observed_rv: repeated_observed_data[observed_rv]
                    for observed_rv in cf.observed_rvs
                }
            },
            augmented_posterior_values=(
                augmented_posterior_values
            ),
            auxiliary_values=auxiliary_values
        )
    )

    loss = tf.reduce_mean(q - p - r)
    losses.append(loss.numpy())

time_3 = time()
print(f"Sampling: {time_3-time_2}")
print(
    "Losses:\n",
    losses,
    "\nMean:\n",
    tf.reduce_mean(losses).numpy(),
    "\nStd:\n",
    tf.math.reduce_std(losses).numpy()
)


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

# %% Storing

base_name = "../data/GM_CF-A_"
pickle.dump(
    samples,
    open(
        base_name + "sample.p",
        "wb"
    )
)
pickle.dump(
    history.history["reverse_KL"],
    open(
        base_name + "history.p",
        "wb"
    )
)
pickle.dump(
    losses,
    open(
        base_name + "losses.p",
        "wb"
    )
)

# %% Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from time import time

from adavi.dual.models import (
    CascadingFlows
)
from generative_hbms.GRE import (
    G, N, scale_mu, scale_mu_g, scale_x,
    cf_hbm_kwargs,
    val_data,
    cf_train_data,
    cf_val_data,
    stack_data
)


# %% CascadingFlows kwargs

d = 16

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

# %% We build our architecture
cf = CascadingFlows(
    **cf_kwargs,
)

# %% We select the loss used for training
cf.compile(
    train_method="reverse_KL",
    n_theta_draws_per_x=32,
    optimizer="adam"
)

# %% We fit the training data
time_1 = time()
history = cf.fit(
    cf_train_data,
    batch_size=32,
    epochs=40,
    shuffle=True,
    verbose=2
)

time_2 = time()
print(f"Training: {time_2 - time_1}")

# %%

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

base_name = "../data/GRE_CF-A_"
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

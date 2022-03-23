# %% Imports
import pickle
from time import time

import tensorflow as tf
import matplotlib.pyplot as plt

from adavi.dual.models import (
    ADAVFamily
)
from generative_hbms.GRE import (
    G, N, scale_mu, scale_mu_g, scale_x,
    hbm_kwargs,
    train_data,
    val_data,
    get_mean_KL_divergence
)


# %% ADAVFamily kwargs

d = 8
num_heads = 2
key_dim = 4
k = 1
m = 8
n_sabs = 2

rff_kwargs = dict(
    units_per_layers=[d]
)

mab_kwargs = dict(
    multi_head_attention_kwargs=dict(
        num_heads=num_heads,
        key_dim=key_dim
    ),
    rff_kwargs=rff_kwargs,
    layer_normalization_h_kwargs=dict(),
    layer_normalization_out_kwargs=dict()
)

isab_kwargs = dict(
    m=m,
    d=d,
    mab_h_kwargs=mab_kwargs,
    mab_out_kwargs=mab_kwargs
)

set_transformer_kwargs = dict(
    embedding_size=d,
    encoder_kwargs=dict(
        type="ISAB",
        kwargs_per_layer=[
            isab_kwargs
        ] * n_sabs
    ),
    decoder_kwargs=dict(
        pma_kwargs=dict(
            k=k,
            d=d,
            rff_kwargs=rff_kwargs,
            mab_kwargs=mab_kwargs,
        ),
        sab_kwargs=mab_kwargs,
        rff_kwargs=rff_kwargs
    )
)

conditional_nf_chain_kwargs = dict(
    nf_type_kwargs_per_bijector=[
        (
            "MAF",
            dict(
                hidden_units=[32, 32, 32]
            )
        ),
        (
            "affine",
            dict(
                scale_type="tril",
                rff_kwargs=rff_kwargs
            )
        )
    ],
    with_permute=False,
    with_batch_norm=False
)

adav_family_kwargs = dict(
    set_transforer_kwargs=set_transformer_kwargs,
    conditional_nf_chain_kwargs=conditional_nf_chain_kwargs,
    **hbm_kwargs
)

# %%

# We build our architecture
adav_family = ADAVFamily(
    **adav_family_kwargs
)

# %% ADAVFamily unregularized ELBO fit
time_1 = time()
adav_family.compile(
    train_method="unregularized_ELBO",
    n_theta_draws_per_x=32,
    optimizer="adam"
)
hist_1 = adav_family.fit(
    train_data,
    batch_size=32,
    epochs=1,  # 10
    shuffle=True,
    verbose=2
)

# %% ADAVFamily reverse KL fit
adav_family.compile(
    train_method="reverse_KL",
    n_theta_draws_per_x=32,
    optimizer="adam"
)
hist_2 = adav_family.fit(
    train_data,
    batch_size=32,
    epochs=1,  # 10
    shuffle=True,
    verbose=2
)

# %%
time_2 = time()
print(f"Training: {time_2 - time_1}")

# %% Sampling

# We store away samples for comparison
n_draws = 1000
samples = {}
losses = []
for val_idx in range(0, 20):
    repeated_x = tf.repeat(
        val_data["x"][val_idx:val_idx + 1],
        repeats=(n_draws,),
        axis=0
    )

    samples[val_idx] = {
        key: value.numpy()
        for key, value in (
            adav_family
            .sample_parameters_conditioned_to_data(
                x=repeated_x
            )
            .items()
        )
    }

    q = adav_family.parameters_log_prob_conditioned_to_data(
        x=repeated_x,
        parameters=samples[val_idx]
    )
    p = adav_family.generative_hbm.log_prob(
        **samples[val_idx],
        x=repeated_x
    )
    loss = tf.reduce_mean(
        q - p
    )
    losses.append(loss.numpy())
time_3 = time()
print(f"Sampling: {time_3-time_2}")
print(
    "Losses:\n",
    losses,
    "\nMean:\n",
    tf.reduce_mean(losses),
    "\nStd:\n",
    tf.math.reduce_std(losses)
)

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

base_name = "../data/GRE_ADAVI_"
pickle.dump(
    samples,
    open(
        base_name + "sample.p",
        "wb"
    )
)
pickle.dump(
    hist_1.history["unregularized_ELBO"]
    + hist_2.history["reverse_KL"],
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
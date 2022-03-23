# %% Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from time import time

from adavi.dual.models import (
    TotalLatentSpaceFlow
)
from generative_hbms.GRE import (
    G, N, scale_mu, scale_mu_g, scale_x,
    total_hbm_kwargs,
    val_data,
    train_data
)
from baselines.summary_networks import (
    SummaryNetwork2Plates
)


# %% Total flow kwargs

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
    ],
    with_permute=False,
    with_batch_norm=False
)

total_kwargs = dict(
    **total_hbm_kwargs,
    conditional_nf_chain_kwargs=conditional_nf_chain_kwargs,
    embedding_size=d * (G + 1)
)

# %% We build our architecture

time_1 = time()
summary_network = SummaryNetwork2Plates(
    set_transformer_kwargs=set_transformer_kwargs
)
model = TotalLatentSpaceFlow(
    **total_kwargs,
    summary_network=summary_network
)

model.compile(
    train_method="forward_KL",
    n_theta_draws_per_x=32,
    optimizer="adam"
)
# %%
history = model.fit(
    train_data,
    batch_size=32,
    epochs=15,
    shuffle=True,
    verbose=2
)

# %%

time_2 = time()
print(f"Training: {time_2 - time_1}")


# %%

samples = {}
losses = []
n_draws = 1_000
for val_idx in range(20):
    repeated_x = tf.repeat(
        val_data["x"][val_idx:val_idx + 1],
        repeats=(n_draws,),
        axis=0
    )
    sample = model.sample_parameters_conditioned_to_data(
        repeated_x
    )
    samples[val_idx] = {
        key: value.numpy()
        for key, value in sample.items()
    }

    p = model.generative_hbm.log_prob(
        **sample,
        **{
            model.observed_rv: repeated_x
        }
    )
    q = (
        model
        .parameters_log_prob_conditioned_to_data(
            parameters=sample,
            x=repeated_x
        )
    )
    losses.append(
        tf.reduce_mean(
            q - p
        )
        .numpy()
    )

time_3 = time()
print(f"Sampling: {time_3 - time_2}")
print(f"losses: {losses}")
print(    
    f"Total G{G} "
    f"mean loss: {tf.reduce_mean(losses).numpy()}, "
    f"std loss: {tf.math.reduce_std(losses).numpy()}"
)

# %%

base_name = "../data/GRE_NPE-C_"
pickle.dump(
    samples,
    open(
        base_name + "sample.p",
        "wb"
    )
)
pickle.dump(
    history.history["forward_KL"],
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


# %%
fig, axs = plt.subplots(
    nrows=5,
    ncols=2,
    figsize=(20, 50)
)
for val_idx in range(0, 5):
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

# %%

# %% Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import os

from time import time

from adavi.dual.models import (
    TotalLatentSpaceFlow
)
from generative_hbms.GM import (
    G, N, scale_mu, scale_mu_g, scale_x,
    total_hbm_kwargs,
    val_data,
    train_data,
    na_val_idx
)
from baselines.summary_networks import (
    SummaryNetwork2Plates
)

# %% Total flow kwargs

d = 16
num_heads = 4
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

samples = {}
losses = []
n_draws = 1_000
val_idx = na_val_idx
summary_network = SummaryNetwork2Plates(
    set_transformer_kwargs=set_transformer_kwargs
)
model = TotalLatentSpaceFlow(
    **total_kwargs,
    summary_network=summary_network
)

time_1 = time()
model.compile(
    train_method="reverse_KL",
    n_theta_draws_per_x=32,
    optimizer="adam"
)
history = model.fit(
    {
        rv: value[val_idx:val_idx + 1]
        for rv, value in val_data.items()
    },
    batch_size=1,
    epochs=1000,
    shuffle=True,
    verbose=2
)

sample = model.sample_parameters_conditioned_to_data(
    tf.repeat(
        val_data["x"][val_idx:val_idx + 1],
        repeats=(n_draws,),
        axis=0
    )
)
samples[val_idx] = {
    key: value.numpy()
    for key, value in sample.items()
}
time_2 = time()

loss = history.history["reverse_KL"][-1]
time = time_2 - time_1

print(
    f"Val idx: {val_idx} time: {time} loss: {loss}"
)

# %% Plotting

fig, axs = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(20, 10)
)
for g in range(G):
    axs[0].scatter(
        val_data["x"][val_idx, g, :, 0],
        val_data["x"][val_idx, g, :, 1],
        color=f"C{g}",
        alpha=0.5
    )

axs[0].axis("equal")
axs[0].set_ylabel(
    f"{val_idx}",
    fontsize=30,
    rotation=0
)
plt.draw()
x_lim = axs[0].get_xlim()
y_lim = axs[0].get_ylim()

axs[1].scatter(
    samples[val_idx]["mu"][:, :, 0],
    samples[val_idx]["mu"][:, :, 1],
    color="black",
    s=20,
    alpha=0.05
)
for g in range(G):
    axs[1].scatter(
        samples[val_idx]["mu_g"][:, g, :, 0],
        samples[val_idx]["mu_g"][:, g, :, 1],
        color=f"C{g}",
        s=20,
        alpha=0.05
    )
axs[1].set_xlim(x_lim)
axs[1].set_ylim(y_lim)

axs[0].set_title(
    "Data",
    fontsize=30
)
axs[1].set_title(
    "Posterior samples",
    fontsize=30
)
plt.show()

# %% Storing

base_name = "../data/GM_TLSF-NA_"
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
    loss,
    open(
        base_name + "losses.p",
        "wb"
    )
)
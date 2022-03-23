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
    na_val_idx,
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

# %% We fit the training data - non amortized
samples = {}
n_draws = 1_000
val_idx = na_val_idx

cf = CascadingFlows(
    **cf_kwargs,
)
cf.compile(
    train_method="reverse_KL",
    n_theta_draws_per_x=32,
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2)
)
time_1 = time()
history = cf.fit(
    {
        key: value[val_idx:val_idx + 1]
        for key, value in cf_val_data.items()
        if key in [
            f"mu_{g}_{n}"
            for g in range(G)
            for n in range(N)
        ]
    },
    batch_size=1,
    epochs=1500,
    shuffle=True,
    verbose=2
)
time_2 = time()
print(f"Training: {time_2 - time_1}")

sample = cf.sample_parameters_conditioned_to_data(
    data={
        f"mu_{g}_{n}": tf.repeat(
            cf_val_data[f"mu_{g}_{n}"][val_idx:val_idx + 1],
            repeats=(n_draws,),
            axis=0
        )
        for g in range(G)
        for n in range(N)
    }
)[0]
samples[val_idx] = stack_data(sample)

loss = history.history["reverse_KL"][-1]
time = time_2 - time_1

print(
    f"Val idx: {val_idx} time: {time} loss: {loss}"
)


# %%

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

# %%

base_name = "../data/GM_CF-NA_"
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

# %%

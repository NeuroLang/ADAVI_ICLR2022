# %% Imports
import pickle

import os
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt

from adavi.dual.models import (
    CascadingFlows
)
from generative_hbms.NC import (
    G,
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
    # train_method="unregularized_ELBO",
    train_method="reverse_KL",
    n_theta_draws_per_x=32,
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2)
)

# %% We fit the training data - amortized
time_1 = time()
history = cf.fit(
    cf_train_data,
    batch_size=32,
    epochs=1,  # 40
    shuffle=True,
    verbose=2
)
time_2 = time()
print(f"Training: {time_2 - time_1}")

# %% Sampling
samples = {}
losses = []
n_draws = 1_000

for val_idx in range(20):
    repeated_observed_data = {
        f"b_{g}": tf.repeat(
            cf_val_data[f"b_{g}"][val_idx:val_idx + 1],
            repeats=(n_draws,),
            axis=0
        )
        for g in range(G)
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

base_name = "../data/NC_CF-A_"

pickle.dump(
    samples,
    open(
        base_name + "sample.p",
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
pickle.dump(
    history.history["reverse_KL"],
    open(
        base_name + "history.p",
        "wb"
    )
)

# %%

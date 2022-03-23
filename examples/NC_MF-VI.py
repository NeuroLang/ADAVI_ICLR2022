# %% Imports
import pickle
import os
from time import process_time

import tensorflow as tf
import tensorflow_probability as tfp

from generative_hbms.NC import (
    D, G,
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

    def log_prob_fn(a):
        return generative_hbm.log_prob(
            a=a,
            b=val_data["b"][val_idx]
        )

    q = tfd.JointDistributionNamed(
        model=dict(
            a=tfd.Independent(
                tfd.Gamma(
                    concentration=tf.Variable(
                        initial_value=tf.ones((D,)),
                        name="q_a_concentration"
                    ),
                    rate=tfp.util.TransformedVariable(
                        initial_value=1.,
                        bijector=tfb.Softplus(),
                        name='q_a_rate'
                    ),
                ),
                reinterpreted_batch_ndims=1,
                name="a"
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
            num_steps=20_000,
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


base_name = "../data/NC_MF-VI_"
pickle.dump(
   samples,
   open(
        base_name+"samples.p",
        "wb"
    )
)

print(
    f"MF VI\n"
    f"times: {times}\n"
    f"mean time: {tf.reduce_mean(times)}\n"
    f"std time: {tf.math.reduce_std(times)}\n"

    f"losses: {losses}\n"
    f"mean loss: {tf.reduce_mean(losses)}\n"
    f"std loss: {tf.math.reduce_std(losses)}"
)

# %%

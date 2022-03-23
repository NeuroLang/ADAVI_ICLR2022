# %% Imports
import pickle
from time import time

import tensorflow as tf

from adavi.dual.models import (
    ADAVFamily
)
from generative_hbms.NC import (
    hbm_kwargs,
    train_data,
    val_data
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

epochs = 40
train_method = "reverse_KL"

# We build our architecture
adav_family = ADAVFamily(
    **adav_family_kwargs
)

# We select the loss used for training
adav_family.compile(
    train_method=train_method,
    n_theta_draws_per_x=32,
    optimizer="adam"
)

# %% Fit
time_1 = time()
# We fit the training data
history = adav_family.fit(
    train_data,
    batch_size=32,
    epochs=epochs,
    shuffle=True,
    verbose=2
)

# %%
time_2 = time()
print(f"Training: {time_2 - time_1}")
# %%

# We store away samples for comparison
n_draws = 1000
samples = {}
losses = []
for val_idx in range(0, 20):
    repeated_x = tf.repeat(
        val_data["b"][val_idx:val_idx + 1],
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
        b=repeated_x
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

# %%

base_name = "../data/NC_ADAVI_"
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

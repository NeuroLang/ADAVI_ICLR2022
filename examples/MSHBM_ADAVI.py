# %% Imports
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

from adavi.dual.models import (
    ADAVFamily
)
from generative_hbms.MSHBM import (
    S, T, L,
    hbm_kwargs,
    train_data,
    val_data
)

# %% ADAVFamily kwargs

d = 32
num_heads = 4
key_dim = 8
k = 1
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

set_transformer_kwargs = dict(
    embedding_size=d,
    encoder_kwargs=dict(
        type="SAB",
        kwargs_per_layer=[
            mab_kwargs
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
                hidden_units=[32],
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0,
                    stddev=1e-4
                ),
                bias_initializer="zeros"
            )
        ),
        (
            "affine",
            dict(
                scale_type="diag",
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

# %% ADAVFamily build

adav_family = ADAVFamily(
    **adav_family_kwargs
)

# To build internal shapes:
sample = adav_family.sample_parameters_conditioned_to_data(
    x=train_data["X_s_t"][0:1]
)
q = adav_family.parameters_log_prob_conditioned_to_data(
    parameters=sample,
    x=train_data["X_s_t"][0:1]
)

# %% ADAVFamily MAP regression
adav_family.compile(
    train_method="exp_MAP_regression",
    n_theta_draws_per_x=32,
    optimizer="adam"
)
adav_family.fit(
    train_data,
    batch_size=32,
    epochs=5,
    shuffle=True
)

# %% ADAVFamily affine unregularized ELBO fit
adav_family.compile(
    train_method="exp_affine_unregularized_ELBO",
    n_theta_draws_per_x=32,
    optimizer="adam"
)
adav_family.fit(
    train_data,
    batch_size=32,
    epochs=1,
    shuffle=True
)

# %% ADAVFamily reverse KL fit
adav_family.compile(
    train_method="reverse_KL",
    n_theta_draws_per_x=32,
    optimizer="adam"
)
adav_family.fit(
    train_data,
    batch_size=32,
    epochs=5,
    shuffle=True
)


# %% Graphical results

n_draws = 100
fig, axs = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(10, 30)
)
for val_idx in range(0, 3):

    axs[val_idx].axis("equal")
    axs[val_idx].set_xlim([-0.1, 1.1])
    axs[val_idx].set_ylim([-0.1, 1.1])
    axs[val_idx].set_ylabel(
        f"{val_idx}",
        fontsize=25,
        rotation=0
    )
    axs[val_idx].set_ylabel(
        f"Example {val_idx}",
        fontsize=30,
        rotation=0,
        ha="right",
        va="center"
    )
    axs[val_idx].tick_params(
        which="major",
        labelsize=30
    )
    axs[val_idx].yaxis.tick_right()
    axs[val_idx].add_patch(
        Arc(
            (0, 0),
            2.0,
            2.0,
            theta1=0.1,
            theta2=89,
            ls="--",
            fill=False,
            color="black",
            alpha=0.5
        )
    )

    sample = adav_family.sample_parameters_conditioned_to_data(
        x=tf.repeat(
            val_data["X_s_t"][val_idx:val_idx + 1],
            repeats=(n_draws,),
            axis=0
        )
    )

    for l in range(L):
        axs[val_idx].plot(
            [
                0.0 * sample["mu_g"][:, l, 0],
                0.77 * sample["mu_g"][:, l, 0]
            ],
            [
                0.0 * sample["mu_g"][:, l, 1],
                0.77 * sample["mu_g"][:, l, 1]
            ],
            color="black",
            alpha=0.05,
        )

    for s in range(S):
        for l in range(L):
            axs[val_idx].plot(
                [
                    0.8 * sample["mu_s"][:, s, l, 0],
                    0.87 * sample["mu_s"][:, s, l, 0]
                ],
                [
                    0.8 * sample["mu_s"][:, s, l, 1],
                    0.87 * sample["mu_s"][:, s, l, 1]
                ],
                color=f"C{T*s}",
                alpha=0.05
            )
        for t in range(T):
            axs[val_idx].scatter(
                x=val_data["X_s_t"][val_idx, s, t, :, 0],
                y=val_data["X_s_t"][val_idx, s, t, :, 1],
                color=f"C{(T+1) * s + t}",
                alpha=0.1,
            )
            for l in range(L):
                axs[val_idx].plot(
                    [
                        0.9 * sample["mu_s_t"][:, s, t, l, 0],
                        0.97 * sample["mu_s_t"][:, s, t, l, 0]
                    ],
                    [
                        0.9 * sample["mu_s_t"][:, s, t, l, 1],
                        0.97 * sample["mu_s_t"][:, s, t, l, 1]
                    ],
                    color=f"C{(T+1) * s + t}",
                    alpha=0.05
                )

axs[0].set_title(
    "Data and posterior sample",
    fontsize=30
)
plt.show()

# %%

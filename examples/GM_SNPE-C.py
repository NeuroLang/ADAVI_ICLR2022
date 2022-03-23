# %% Imports
from typing import Tuple, Dict
import pickle
import os
from time import process_time

import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from sbi import (
    utils as sbi_utils,
    inference as sbi_inference
)
from generative_hbms.GM import (
    G, N, D,
    generative_hbm,
    generative_prior,
    dataset,
    na_val_idx
)
from juho_lee_set_transformer.modules import (
    SAB,
    PMA
)

tfd = tfp.distributions


# %% Global variables
input_key = "x"
theta_shapes = {
    key: shape.as_list()
    for key, shape in (
        generative_hbm
        .event_shape
        .items()
    )
    if key != input_key
}
val_idx = na_val_idx
n_draws = 1000
num_rounds = 5
samples_per_round = 1000

# %% IO related to theta


def flatten_theta(
    theta_dict: Dict,
    batch_shape: Tuple
) -> torch.Tensor:
    """Converts a dictof tensors
    into one big tensor

    Parameters
    ----------
    theta_dict : Dict
        dict of Numpy tensors
        shapes: {
            key:
                batch_shape
                + Shape(key)
        }
    batch_shape : Tuple
        necessary for reshaping

    Returns
    -------
    torch.Tensor
        one big tensor
        shape:
            batch_shape
            + (-1,)
    """
    return torch.cat(
        [
            torch.from_numpy(
                value
            )
            .reshape(
                batch_shape + (-1,)
            )
            for key, value in theta_dict.items()
            if key != input_key
        ],
        axis=-1
    )


def unflatten_theta(
    theta_flat: torch.Tensor,
    theta_shapes: Dict
) -> Dict:
    """Takes one big tensor and
    splits into a dict of tensors

    Parameters
    ----------
    theta_flat : torch.Tensor
        one big tensor
        shape:
            batch_shape
            + (-1,)
    theta_shapes : Dict
        dict of shapes: {
            key: Shape(key)
        }

    Returns
    -------
    Dict
        dict of Torch tensors
        shapes: {
            key:
                batch_shape
                + Shape(key)
        }
    """
    index = 0
    output = {}
    for key, shape in theta_shapes.items():
        if key == input_key:
            continue
        size = np.prod(shape)
        output[key] = (
            theta_flat[:, index:index + size]
            .reshape(
                [-1] + shape
            )
        )
        index += size
    return output


class FlatPrior():
    def __init__(
        self,
        generative_prior: tfd.JointDistributionNamed
    ):
        """Wrap-up for tfp generative prior

        Parameters
        ----------
        generative_prior : tfd.JointDistributionNamed
            tfp generative prior
        """
        self.generative_prior = generative_prior

    def sample(
        self,
        batch_shape: Tuple,
        **kwargs
    ) -> torch.Tensor:
        """Samples from inner generative prior
        and flattens the result

        Parameters
        ----------
        batch_shape : Tuple
            size of the sample

        Returns
        -------
        torch.Tensor
            one big tensor
            shape: batch_shape + (-1,)
        """
        sample = self.generative_prior.sample(batch_shape)
        return flatten_theta(
            theta_dict={
                key: value.numpy()
                for key, value in sample.items()
            },
            batch_shape=batch_shape
        )

    def log_prob(
        self,
        theta_flat: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Takes in a flat theta tensor,
        unflattens it and calls inner
        generative prior

        Parameters
        ----------
        theta_flat : torch.Tensor
            one big tensor
            shape: batch_shape + (-1,)

        Returns
        -------
        torch.Tensor
            log prob tensor
            shape: batch_shape
        """
        if len(theta_flat.shape) == 1:
            theta_flat = theta_flat.unsqueeze(0)
        theta_dict = unflatten_theta(
            theta_flat=theta_flat,
            theta_shapes=theta_shapes
        )

        log_prob = self.generative_prior.log_prob(
            **{
                key: value.numpy()
                for key, value in theta_dict.items()
            }
        ).numpy()

        return torch.from_numpy(log_prob)

# %% Embedder for high-rank data


class GroupPopEncoder(nn.Module):
    def __init__(self):
        """Encoder based on Set Transformer (Lee et al. 2020)
        Contracts the N plate in parallel for G groups
        to obtain group statistics, then the G plate
        for ppopulation statistics
        """
        super().__init__()

        self.enc1 = nn.Sequential(
            SAB(dim_in=2, dim_out=8, num_heads=4),
            SAB(dim_in=8, dim_out=8, num_heads=4),
        )
        self.dec1 = nn.Sequential(
            PMA(dim=8, num_heads=4, num_seeds=1),
            nn.Linear(in_features=8, out_features=8),
        )

        self.enc2 = nn.Sequential(
            SAB(dim_in=8, dim_out=8, num_heads=4),
            SAB(dim_in=8, dim_out=8, num_heads=4),
        )
        self.dec2 = nn.Sequential(
            PMA(dim=8, num_heads=4, num_seeds=1),
            nn.Linear(in_features=8, out_features=8),
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Contracts the N plate in parallel for G groups
        to obtain group statistics, then the G plate
        for ppopulation statistics

        Parameters
        ----------
        x : torch.Tensor
            shape:
                (batch_size,)
                + (G, N, D)

        Returns
        -------
        torch.Tensor
            shape:
                (batch_size,)
                + ((G + 1) * 8,)
        """
        batch_size = x.shape[0]
        z = x.reshape((batch_size * G, N, D))
        z = self.enc1(z)
        z = self.dec1(z)

        z1 = z.reshape((batch_size, G * 8))

        z2 = z.reshape((batch_size, G, 8))
        z2 = self.enc2(z2)
        z2 = self.dec2(z2)
        z2 = z2.reshape((batch_size, 8))

        z3 = torch.cat(
            [
                z1,
                z2
            ],
            axis=-1
        )

        return z3
# %% flat data


flat_dataset = {
    key: {
        "x": torch.from_numpy(
            dataset[key]["x"]
        ),
        "theta": flatten_theta(
            theta_dict=dataset[key],
            batch_shape=(
                dataset[key]["x"].shape[0],
            )
        )
    }
    for key in ["train", "val"]
}
prior = FlatPrior(
    generative_prior=generative_prior
)

# %%
# SNPE-C training
# ---------------

embedding_net = GroupPopEncoder()
density_estimator = sbi_utils.get_nn_models.posterior_nn(
    model="maf",
    embedding_net=embedding_net
)
snpe = sbi_inference.SNPE(
    prior=prior,
    density_estimator=density_estimator
)

datum = flat_dataset["val"]["x"][val_idx:val_idx + 1]
posteriors = []
proposal = prior

for _ in range(num_rounds):
    theta_flat = proposal.sample(
        (samples_per_round,),
        x=datum
    )

    theta_dict = unflatten_theta(
        theta_flat=theta_flat,
        theta_shapes=theta_shapes
    )
    theta_dict["probs"] = nn.Softmax(dim=-1)(theta_dict["probs"])

    X = generative_hbm.experimental_pin(
        **{
            key: value.numpy()
            for key, value in theta_dict.items()
        }
    ).sample_unpinned()["x"].numpy()
    X = torch.from_numpy(X)

    density_estimator_train = (
        snpe
        .append_simulations(
            theta_flat,
            X,
            proposal=proposal
        )
        .train()
    )
    posterior = snpe.build_posterior(density_estimator_train)
    posteriors.append(posterior)
    proposal = posterior.set_default_x(datum)

posterior = posteriors[-1]
theta_flat = posterior.sample(
    (n_draws,),
    x=datum
)

theta = unflatten_theta(
    theta_flat=theta_flat,
    theta_shapes=theta_shapes
)
theta["probs"] = nn.Softmax(dim=-1)(theta_dict["probs"])

samples = {}
samples[val_idx] = {
    rv: value.numpy()
    for rv, value in theta.items()
}

q = posterior.log_prob(theta_flat)
p = generative_hbm.log_prob(
    x=datum,
    **samples[val_idx]
)
loss = tf.reduce_mean(q - p)

print(f"Val idx: {val_idx}, loss: {loss}")

# %% Storing

base_name = "../data/GM_SNPE-C_"
pickle.dump(
    samples,
    open(
        base_name + "sample.p",
        "wb"
    )
)
pickle.dump(
    loss,
    open(
        base_name + "loss.p",
        "wb"
    )
)
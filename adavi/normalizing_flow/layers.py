from typing import List, Dict

import tensorflow as tf

from tensorflow.keras.layers import (
    Dense
)


class ConditionalStateTimeDerivative(tf.keras.layers.Layer):
    def __init__(
        self,
        units_per_layers: List[int],
        output_size: int,
        dense_kwargs: Dict = dict(activation="tanh"),
        **kwargs
    ):
        """Conditional state time derivative layer.
        Used for a conditional implementation of FFJORD

        Parameters
        ----------
        units_per_layers : List[int]
            units per hidden layer
        output_size : int
            units for the last layer (no activation)
        dense_kwargs : Dict, optional
            kwargs for hidden layers, by default dict(activation="tanh")
        """
        super().__init__(**kwargs)

        self.mlp = tf.keras.Sequential(
            layers=(
                [
                    Dense(
                        units=units,
                        **dense_kwargs
                    )
                    for units in units_per_layers
                ]
                +
                [
                    Dense(
                        units=output_size,
                    )
                ]
            )
        )

    def __call__(
        self,
        time: tf.Tensor,
        state: tf.Tensor,
        conditional_input: tf.Tensor
    ) -> tf.Tensor:
        """concatenates all inputs and calls an MLP

        Parameters
        ----------
        time : tf.Tensor
            shape (1,)
        state : tf.Tensor
            shape: batch_shape + (state_size,)
        conditional_input : tf.Tensor
            shape: batch_shape + (conditional_size)

        Returns
        -------
        tf.Tensor
            derivatives for the state
            shape: batch_shape + (state_size,)
        """
        inputs = tf.concat(
            [
                tf.broadcast_to(
                    time,
                    state.shape
                ),
                state,
                conditional_input
            ],
            axis=-1
        )
        return self.mlp(inputs)

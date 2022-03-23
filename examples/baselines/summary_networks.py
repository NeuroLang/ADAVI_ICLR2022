from typing import Dict

import tensorflow as tf

from adavi.set_transformer.models import SetTransformer


class SummaryNetwork2Plates(tf.keras.Model):

    def __init__(
        self,
        set_transformer_kwargs: Dict,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.ST1 = SetTransformer(
            attention_axes=(-3,),
            **set_transformer_kwargs,
            name="ST1"
        )

        self.ST2 = SetTransformer(
            attention_axes=(-3,),
            **set_transformer_kwargs,
            name="ST2"
        )

    def call(
        self,
        x: tf.Tensor
    ) -> tf.Tensor:
        B, G, N, D = x.shape
        z1 = self.ST1(x)
        z1 = tf.squeeze(z1, axis=-2)

        z2 = self.ST2(z1)
        z2 = tf.squeeze(z2, axis=-2)

        y = tf.concat(
            [
                tf.reshape(
                    z1,
                    (B, -1)
                ),
                z2
            ],
            axis=-1
        )

        return y

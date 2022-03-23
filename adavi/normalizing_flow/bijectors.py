from typing import List, Dict, Tuple, Literal

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import (
    Dense
)
from ..set_transformer.layers import (
    RFF
)

tfd = tfp.distributions
tfb = tfp.bijectors


class BatchedHighwayFlow(tfp.experimental.bijectors.HighwayFlow):
    """Modifies the TFP implem to allow for batching"""
    def _gated_residual_fraction(self):
        """Returns a vector of residual fractions that
        encodes gated dimensions."""
        return self.residual_fraction * tf.concat(
            [
                tf.ones([self.gate_first_n], dtype=self.dtype),
                tf.zeros([self.num_ungated], dtype=self.dtype)
            ],
            axis=0
        )


class ConditionalHighwayFlow(tfb.Bijector):

    def __init__(
        self,
        rff_kwargs: Dict,
        event_size: int,
        **kwargs
    ):
        """Highway Flow where all weights are functions of a
        conditional input

        Parameters
        ----------
        rff_kwargs : Dict
            for the weights regressors
        event_size : int
            from the base distribution
        """
        super().__init__(
            forward_min_event_ndims=1,
            **kwargs
        )

        self.lower_bijector = tfb.Chain(
            [
                tfb.TransformDiagonal(diag_bijector=tfb.Shift(1.)),
                tfb.Pad(paddings=[(1, 0), (0, 1)]),
                tfb.FillTriangular(),
            ]
        )
        lower_size = self.lower_bijector.inverse_event_shape(
            [event_size, event_size]
        )[0]
        self.lower_rff = tf.keras.Sequential(
            layers=[
                RFF(**rff_kwargs),
                Dense(
                    units=lower_size,
                    kernel_initializer=(
                        tf.keras.initializers
                        .RandomNormal(
                            mean=0.0,
                            stddev=1e-4
                        )
                    ),
                    bias_initializer="zeros"
                )
            ],
        )

        self.upper_bijector = tfb.FillScaleTriL(
            diag_bijector=tfb.Softplus(),
            diag_shift=None
        )
        upper_size = self.upper_bijector.inverse_event_shape(
            [event_size, event_size]
        )[0]
        self.upper_rff = tf.keras.Sequential(
            layers=[
                RFF(**rff_kwargs),
                Dense(
                    units=upper_size,
                    kernel_initializer=(
                        tf.keras.initializers
                        .RandomNormal(
                            mean=0.0,
                            stddev=1e-4
                        )
                    ),
                    bias_initializer="zeros"
                )
            ],
        )
        self.bias_rff = tf.keras.Sequential(
            layers=[
                RFF(**rff_kwargs),
                Dense(
                    units=event_size,
                    kernel_initializer=(
                        tf.keras.initializers
                        .RandomNormal(
                            mean=0.0,
                            stddev=1e-4
                        )
                    ),
                    bias_initializer="zeros"
                )
            ],
        )

    def forward(
        self,
        x: tf.Tensor,
        conditional_input: tf.Tensor,
        residual_fraction: tf.Tensor,
        **kwargs
    ) -> tf.Tensor:
        """HighwayFlow(...).forward(x)

        Parameters
        ----------
        x : tf.Tensor
            shape: batch_shape + (event_size,)
        conditional_input : tf.Tensor
            shape: batch_shape + (conditional_size,)
        residual_fraction
            shape: batch_shape + (1,)

        Returns
        -------
        tf.Tensor
            shape: batch_shape + (event_size,)
        """
        return BatchedHighwayFlow(
            residual_fraction=residual_fraction,
            activation_fn=tf.nn.softplus,
            bias=self.bias_rff(conditional_input),
            upper_diagonal_weights_matrix=self.upper_bijector(
                self.upper_rff(
                    conditional_input
                )
            ),
            lower_diagonal_weights_matrix=self.lower_bijector(
                self.lower_rff(
                    conditional_input
                )
            ),
            gate_first_n=None
        ).forward(x)

    def inverse(
        self,
        y: tf.Tensor,
        conditional_input: tf.Tensor,
        residual_fraction: tf.Tensor,
        **kwargs
    ) -> tf.Tensor:
        """HighwayFlow(...).inverse(y)

        Parameters
        ----------
        y : tf.Tensor
            shape: batch_shape + (event_size,)
        conditional_input : tf.Tensor
            shape: batch_shape + (conditional_size,)
        residual_fraction
            shape: batch_shape + (1,)

        Returns
        -------
        tf.Tensor
            shape: batch_shape + (event_size,)
        """
        return BatchedHighwayFlow(
            residual_fraction=residual_fraction,
            activation_fn=tf.nn.softplus,
            bias=self.bias_rff(conditional_input),
            upper_diagonal_weights_matrix=self.upper_bijector(
                self.upper_rff(
                    conditional_input
                )
            ),
            lower_diagonal_weights_matrix=self.lower_bijector(
                self.lower_rff(
                    conditional_input
                )
            ),
            gate_first_n=None
        ).inverse(y)

    def inverse_log_det_jacobian(
        self,
        y: tf.Tensor,
        event_ndims: int,
        conditional_input: tf.Tensor,
        residual_fraction: tf.Tensor,
        **kwargs
    ) -> tf.Tensor:
        """HighwayFlow(...).inverse_log_det_jacobian(y, event_ndims)

        Parameters
        ----------
        y : tf.Tensor
            shape: batch_shape + (event_size,)
        event_ndims : int
            rank of event shape (i.e. 1 here)
        conditional_input : tf.Tensor
            shape: batch_shape + (conditional_size,)
        residual_fraction
            shape: batch_shape + (1,)

        Returns
        -------
        tf.Tensor
            shape: batch_shape
        """
        return BatchedHighwayFlow(
            residual_fraction=residual_fraction,
            activation_fn=tf.nn.softplus,
            bias=self.bias_rff(conditional_input),
            upper_diagonal_weights_matrix=self.upper_bijector(
                self.upper_rff(
                    conditional_input
                )
            ),
            lower_diagonal_weights_matrix=self.lower_bijector(
                self.lower_rff(
                    conditional_input
                )
            ),
            gate_first_n=None
        ).inverse_log_det_jacobian(
            y=y,
            event_ndims=event_ndims
        )


class ConditionalAffine(tfb.Bijector):

    def __init__(
        self,
        scale_type: Literal["diag", "tril", "none"],
        rff_kwargs: Dict,
        event_size: int,
        **kwargs
    ):
        """Affine transform where the shift and scale
        are functions of a conditional input

        Parameters
        ----------
        scale_type : Literal["diag", "tril", "none]
            determines parametrization of the affine bijector
        rff_kwargs : Dict
            for shift and scale regressors
        event_size : int
            from the base distribution
        """
        super().__init__(
            forward_min_event_ndims=1,
            **kwargs
        )

        if scale_type not in [
            "diag",
            "tril",
            "none"
        ]:
            raise NotImplementedError(
                f"unknown scale type {scale_type}"
            )

        self.scale_type = scale_type

        self.shift = tf.keras.Sequential(
            layers=[
                RFF(**rff_kwargs),
                Dense(units=event_size)
            ],
        )
        if self.scale_type != "none":
            self.scale = tf.keras.Sequential(
                layers=[
                    RFF(**rff_kwargs),
                    Dense(
                        units=(
                            event_size
                            if self.scale_type == "diag"
                            else
                            (event_size * (event_size + 1)) / 2
                        )
                    )
                ],
            )

    def forward(
        self,
        x: tf.Tensor,
        conditional_input: tf.Tensor,
        **kwargs
    ) -> tf.Tensor:
        """Affine(
            shift=f(conditional_input),
            scale=g(conditional_input)
        ).forward(x)

        Parameters
        ----------
        x : tf.Tensor
            shape: batch_shape + (event_size,)
        conditional_input : tf.Tensor
            shape: batch_shape + (conditional_size,)

        Returns
        -------
        tf.Tensor
            shape: batch_shape + (event_size,)
        """
        return tfb.Chain(
            [
                tfb.Shift(
                    shift=self.shift(conditional_input)
                ),
            ]
            +
            (
                [
                    tfb.ScaleMatvecDiag(
                        scale_diag=self.scale(conditional_input)
                    )
                ]
                if self.scale_type == "diag"
                else
                [
                    tfb.ScaleMatvecTriL(
                        scale_tril=tfp.math.fill_triangular(
                            self.scale(conditional_input)
                        )
                    )
                ]
                if self.scale_type == "tril"
                else
                []
            )
        ).forward(x)

    def inverse(
        self,
        y: tf.Tensor,
        conditional_input: tf.Tensor,
        **kwargs
    ) -> tf.Tensor:
        """Affine(
            shift=f(conditional_input),
            scale=g(conditional_input)
        ).inverse(y)

        Parameters
        ----------
        y : tf.Tensor
            shape: batch_shape + (event_size,)
        conditional_input : tf.Tensor
            shape: batch_shape + (conditional_size,)

        Returns
        -------
        tf.Tensor
            shape: batch_shape + (event_size,)
        """
        return tfb.Chain(
            [
                tfb.Shift(
                    shift=self.shift(conditional_input)
                ),
            ]
            +
            (
                [
                    tfb.ScaleMatvecDiag(
                        scale_diag=self.scale(conditional_input)
                    )
                ]
                if self.scale_type == "diag"
                else
                [
                    tfb.ScaleMatvecTriL(
                        scale_tril=tfp.math.fill_triangular(
                            self.scale(conditional_input)
                        )
                    )
                ]
                if self.scale_type == "tril"
                else
                []
            )
        ).inverse(y)

    def inverse_log_det_jacobian(
        self,
        y: tf.Tensor,
        event_ndims: int,
        conditional_input: tf.Tensor,
        **kwargs
    ) -> tf.Tensor:
        """Affine(
            shift=f(conditional_input),
            scale=g(conditional_input)
        ).forward(x)

        Parameters
        ----------
        y : tf.Tensor
            shape: batch_shape + (event_size,)
        event_ndims : int
            rank of event shape (i.e. 1 here)
        conditional_input : tf.Tensor
            shape: batch_shape + (conditional_size,)

        Returns
        -------
        tf.Tensor
            shape: batch_shape
        """
        return tfb.Chain(
            [
                tfb.Shift(
                    shift=self.shift(conditional_input)
                ),
            ]
            +
            (
                [
                    tfb.ScaleMatvecDiag(
                        scale_diag=self.scale(conditional_input)
                    )
                ]
                if self.scale_type == "diag"
                else
                [
                    tfb.ScaleMatvecTriL(
                        scale_tril=tfp.math.fill_triangular(
                            self.scale(conditional_input)
                        )
                    )
                ]
                if self.scale_type == "tril"
                else
                []
            )
        ).inverse_log_det_jacobian(
            y=y,
            event_ndims=event_ndims
        )


class ConditionalNFChain(tfb.Bijector):
    def __init__(
        self,
        nf_type_kwargs_per_bijector: List[
            Tuple[
                Literal["MAF", "affine", "realNVP", "Highway"],
                Dict
            ]
        ],
        event_size: int,
        conditional_event_size: int,
        with_permute: bool = True,
        with_batch_norm: bool = True,
        **kwargs
    ):
        """Stacks multiple NF bijectors of various types
        will broadcast conditional input to those

        Parameters
        ----------
        nf_type_kwargs_per_bijector : List[
            Tuple[Literal["MAF", "affine", "realNVP", "Highway"], Dict]
        ]
            list of (nf_type, nf_kwargs)
            determines number, type and parametrization
            of conditional NF bijectors in the chain
        event_size : int
            from base distribution
        conditional_event_size : int
            from upstream encoding
        with_permute : bool, optional
            permute event tensor between conditional NF bijectors,
            by default True
        with_batch_norm : bool, optional
            normalize event tensor between conditional NF bijectors,
            by default True
        """
        super().__init__(
            forward_min_event_ndims=1,
            **kwargs
        )
        self.event_size = event_size
        self.conditional_event_size = conditional_event_size
        self.regressors = {}

        self.conditional_bijectors = {}
        self.bijectors = []
        for (nf_type, nf_kwargs) in nf_type_kwargs_per_bijector:
            if nf_type == "Highway":
                self.residual_fraction_rff = tf.keras.Sequential(
                    layers=[
                        RFF(**nf_kwargs["rff_kwargs"]),
                        Dense(
                            units=1,
                            activation="sigmoid",
                            kernel_initializer=(
                                tf.keras.initializers
                                .RandomNormal(
                                    mean=0.0,
                                    stddev=1e-4
                                )
                            ),
                            bias_initializer=(
                                tf.keras.initializers
                                .Constant(
                                    value=0.5
                                )
                            )
                        )
                    ],
                )
                continue

        for b, (nf_type, nf_kwargs) in enumerate(
            nf_type_kwargs_per_bijector
        ):
            if with_permute:
                self.bijectors.append(
                    tfb.Permute(
                        permutation=(
                            np
                            .random
                            .permutation(self.event_size,)
                        )
                    )
                )
            if with_batch_norm:
                self.bijectors.append(
                    tfb.Invert(
                        tfb.BatchNormalization()
                    )
                )
            name, bijector = self.build_bijector(
                b=b,
                nf_type=nf_type,
                nf_kwargs=nf_kwargs
            )
            self.conditional_bijectors[nf_type, name] = bijector
            self.bijectors.append(bijector)

        self.chain = tfb.Chain(self.bijectors)

    def build_bijector(
        self,
        b: int,
        nf_type: Literal["MAF", "affine", "realNVP", "Highway"],
        nf_kwargs: Dict,
        **kwargs
    ) -> Tuple[str, tfb.Bijector]:
        """Returns a conditional NF bijector
        of given type

        Parameters
        ----------
        b : int
            used for naming
        nf_type : Literal["MAF", "affine", "realNVP", "Highway"]
            determines type of conditional NF bijector
        nf_kwargs : Dict
            dependent on the NF type

        Returns
        -------
        Tuple[str, tfb.Bijector]
            tuple (name : str, nf bijector : tfb.Bijector)

        Raises
        ------
        NotImplementedError
            if nf_type not in ["MAF", "affine", "realNVP", "Highway"]
        """
        if nf_type == "MAF":
            made = tfb.AutoregressiveNetwork(
                params=2,
                event_shape=self.event_size,
                conditional=True,
                conditional_event_shape=(self.conditional_event_size,),
                **nf_kwargs
            )
            name = f"MAF_{b}"

            return (
                name,
                tfb.MaskedAutoregressiveFlow(
                    made,
                    name=name
                )
            )
        elif nf_type == "affine":
            name = f"affine_{b}"
            return (
                name,
                ConditionalAffine(
                    event_size=self.event_size,
                    name=name,
                    **nf_kwargs
                )
            )
        elif nf_type == "realNVP":
            name = f"realNVP_{b}"
            for regressor in ["shift", "log_scale"]:
                self.regressors[f"{name}_{regressor}"] = (
                    tf.keras.Sequential(
                        layers=[
                            RFF(**nf_kwargs),
                            Dense(
                                units=(
                                    self.event_size
                                    -
                                    self.event_size // 2
                                )
                            )
                        ],
                    )
                )

            def shift_and_log_scale_fn(
                x: tf.Tensor,
                input_depth: int,
                conditional_input: tf.Tensor
            ) -> Tuple[tf.Tensor, tf.Tensor]:
                z = tf.concat(
                    [
                        x,
                        conditional_input
                    ],
                    axis=-1
                )
                return (
                    self.regressors[f"{name}_shift"](z),
                    self.regressors[f"{name}_log_scale"](z)
                )

            return (
                name,
                tfb.RealNVP(
                    num_masked=self.event_size // 2,
                    shift_and_log_scale_fn=shift_and_log_scale_fn,
                    name=name
                )
            )
        elif nf_type == "Highway":
            name = f"highway_{b}"
            return (
                name,
                ConditionalHighwayFlow(
                    event_size=self.event_size,
                    name=name,
                    **nf_kwargs
                )
            )
        else:
            raise NotImplementedError(
                f"{nf_type} is not a valid NF type"
            )

    def forward(
        self,
        chain_input: tf.Tensor,
        conditional_input: tf.Tensor,
        **kwargs
    ) -> tf.Tensor:
        """Transforms chain_input, broadcasting
        conditional_input to all conditional NF bijectors

        Parameters
        ----------
        chain_input : tf.Tensor
            from base distribution
            shape: batch_shape + (event_size,)
        conditional_input : tf.Tensor
            from upstream encoding
            shape: batch_shape + (conditional_event_size,)

        Returns
        -------
        tf.Tensor
            transformed chain input
            shape: batch_shape + (event_size,)
        """
        return self.chain.forward(
            x=chain_input,
            **{
                key: (
                    {
                        "conditional_input": conditional_input,
                        "residual_fraction": self.residual_fraction_rff(
                            conditional_input
                        )
                    }
                    if nf_type == "Highway"
                    else
                    {
                        "conditional_input": conditional_input
                    }
                )
                for nf_type, key in (
                    self
                    .conditional_bijectors
                    .keys()
                )
            },
            **kwargs
        )

    def inverse(
        self,
        chain_input: tf.Tensor,
        conditional_input: tf.Tensor,
        **kwargs
    ) -> tf.Tensor:
        """Transforms chain_input, broadcasting
        conditional_input to all conditional NF bijectors

        Parameters
        ----------
        chain_input : tf.Tensor
            from base distribution
            shape: batch_shape + (event_size,)
        conditional_input : tf.Tensor
            from upstream encoding
            shape: batch_shape + (conditional_event_size,)

        Returns
        -------
        tf.Tensor
            transformed chain input
            shape: batch_shape + (event_size,)
        """
        return self.chain.inverse(
            y=chain_input,
            **{
                key: (
                    {
                        "conditional_input": conditional_input,
                        "residual_fraction": self.residual_fraction_rff(
                            conditional_input
                        )
                    }
                    if nf_type == "Highway"
                    else
                    {
                        "conditional_input": conditional_input
                    }
                )
                for nf_type, key in (
                    self
                    .conditional_bijectors
                    .keys()
                )
            },
            **kwargs
        )

    def inverse_log_det_jacobian(
        self,
        chain_input: tf.Tensor,
        event_ndims: int,
        conditional_input: tf.Tensor,
        **kwargs
    ) -> tf.Tensor:
        """Retruns inverse log det Jacobian at chain_input,
        broadcasting conditional_input to all conditional NF bijectors

        Parameters
        ----------
        chain_input : tf.Tensor
            from base distribution
            shape: batch_shape + (event_size,)
        event_ndims : int
            rank of event shape (i.e. 1 here)
        conditional_input : tf.Tensor
            from upstream encoding
            shape: batch_shape + (conditional_event_size,)

        Returns
        -------
        tf.Tensor
            shape: batch_shape
        """
        return self.chain.inverse_log_det_jacobian(
            y=chain_input,
            event_ndims=event_ndims,
            **{
                key: (
                    {
                        "conditional_input": conditional_input,
                        "residual_fraction": self.residual_fraction_rff(
                            conditional_input
                        )
                    }
                    if nf_type == "Highway"
                    else
                    {
                        "conditional_input": conditional_input
                    }
                )
                for nf_type, key in (
                    self
                    .conditional_bijectors
                    .keys()
                )
            },
            **kwargs
        )

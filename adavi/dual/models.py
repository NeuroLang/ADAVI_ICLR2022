from typing import Dict, Union, Literal, Tuple, List, Optional
from functools import partial
from collections import defaultdict

import networkx as nx
import tensorflow as tf
import tensorflow_probability as tfp

from networkx.algorithms import tree
from ..set_transformer.layers import (
    RFF
)
from ..set_transformer.models import (
    SetTransformer
)
from ..normalizing_flow.bijectors import (
    ConditionalAffine,
    ConditionalNFChain
)
from .graph import (
    get_inference_order
)

tfd = tfp.distributions
tfb = tfp.bijectors


class TotalLatentSpaceFlow(tf.keras.Model):

    def __init__(
        self,
        generative_hbm: tfd.JointDistributionNamed,
        link_functions: Dict[str, tfb.Bijector],
        observed_rv: str,
        conditional_nf_chain_kwargs: Dict,
        amortized: bool = False,
        summary_network: Optional[tf.keras.Model] = None,
        embedding_size: Optional[int] = None,
        **kwargs
    ):
        """Performs density estimation using a single NF over the total
        latent space (TLSF)

        Parameters
        ----------
        generative_hbm : tfd.JointDistributionNamed
            HBM to perform inference upon
        link_functions : Dict[str, tfb.Bijector]
            dict {rv: link_function}
        observed_rv : str
            key for the observed RV in the generative HBM
        conditional_nf_chain_kwargs : Dict
            ConditionalNFChain kwargs
        amortized : bool, optional
            determines if TLSF is amortized,
            by default False
        summary_network : Optional[tf.keras.Model], optional
            used if amortized, encodes observed RV's data,
            by default None
        embedding_size : Optional[int], optional
            used if amortized,
            output size for the summary network,
            by default None
        """
        super().__init__(**kwargs)

        self.generative_hbm = generative_hbm
        self.link_functions = link_functions
        self.observed_rv = observed_rv

        self.summary_network = summary_network
        self.embedding_size = embedding_size
        self.build_architecture(
            conditional_nf_chain_kwargs=conditional_nf_chain_kwargs
        )

    def build_architecture(
        self,
        conditional_nf_chain_kwargs: Dict
    ) -> None:
        """Build the single NF in the architecture

        Parameters
        ----------
        conditional_nf_chain_kwargs : Dict
            ConditionalNFChain kwargs
        """
        self.latent_rvs = [
            rv
            for rv in self.generative_hbm.event_shape.keys()
            if rv != self.observed_rv
        ]

        constrainers = {}
        event_sizes = {}
        for rv in self.latent_rvs:
            shape = (
                self
                .generative_hbm
                .event_shape
                [rv]
            )
            constrained_shape = (
                self
                .link_functions[rv]
                .inverse_event_shape(
                    shape
                )
            )
            event_size = tf.reduce_prod(
                constrained_shape
            )
            event_sizes[rv] = event_size
            reshaper = tfb.Reshape(
                event_shape_in=(event_size,),
                event_shape_out=constrained_shape
            )
            constrainers[rv] = tfb.Chain(
                [
                    self.link_functions[rv],
                    reshaper
                ]
            )

        self.splitter = tfb.Split(
            [
                event_sizes[rv]
                for rv in self.latent_rvs
            ]
        )

        self.restructurer = tfb.Restructure(
            {
                rv: rank
                for rank, rv in enumerate(
                    self.latent_rvs
                )
            }
        )

        self.constrainer = tfb.JointMap(
            constrainers
        )

        total_event_size = int(
            tf.reduce_sum(
                [
                    event_sizes[rv]
                    for rv in self.latent_rvs
                ]
            )
        )
        self.base_dist = tfd.Independent(
            tfd.Normal(
                loc=tf.zeros((total_event_size,)),
                scale=1
            ),
            reinterpreted_batch_ndims=1
        )

        self.conditional_nf_chain = ConditionalNFChain(
            **conditional_nf_chain_kwargs,
            event_size=total_event_size,
            conditional_event_size=self.embedding_size,
            name="nf"
        )

        self.transformed_dist = tfd.TransformedDistribution(
            self.base_dist,
            bijector=tfb.Chain(
                [
                    self.constrainer,
                    self.restructurer,
                    self.splitter,
                    self.conditional_nf_chain
                ]
            )
        )

    def encode_data(
        self,
        x: tf.Tensor
    ) -> tf.Tensor:
        """Encodes data (used if amortized)

        Parameters
        ----------
        x : tf.Tensor
            observed RV's data

        Returns
        -------
        tf.Tensor
            embedding
        """
        return self.summary_network(x)

    def sample_parameters_conditioned_to_encoding(
        self,
        encoding: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Samples from the latent parameters conditional
        to the encoding

        Parameters
        ----------
        encoding : tf.Tensor
            output from summary network

        Returns
        -------
        Dict[str, tf.Tensor]
            dict {rv: value}
        """
        batch_size = encoding.shape[0]

        return self.transformed_dist.sample(
            (batch_size,),
            bijector_kwargs=dict(
                nf=dict(
                    conditional_input=encoding
                )
            )
        )

    def sample_parameters_conditioned_to_data(
        self,
        x: tf.Tensor
    ) -> Dict:
        """Wrapper that encodes data and then samples
        from the latent parameters

        Parameters
        ----------
        x : tf.Tensor
            observed RV's data

        Returns
        -------
        Dict
            dict {rv: value}
        """
        encoding = self.encode_data(x)
        return self.sample_parameters_conditioned_to_encoding(encoding)

    def parameters_log_prob_conditioned_to_encoding(
        self,
        parameters: Dict[str, tf.Tensor],
        encoding: tf.Tensor
    ) -> tf.Tensor:
        """Log prob of parameters conditional on the
        fed encoding

        Parameters
        ----------
        parameters : Dict[str, tf.Tensor]
            dict {rv: value}
        encoding : tf.Tensor
            output from summary network

        Returns
        -------
        tf.Tensor
            log prob tensor
        """
        return self.transformed_dist.log_prob(
            parameters,
            bijector_kwargs=dict(
                nf=dict(
                    conditional_input=encoding
                )
            )
        )

    def parameters_log_prob_conditioned_to_data(
        self,
        parameters: Dict[str, tf.Tensor],
        x: tf.Tensor
    ) -> tf.Tensor:
        """Wrapper that encodes data and then computes log prob

        Parameters
        ----------
        parameters : Dict[str, thf.Tensor]
            dict {rv: value}
        x : tf.Tensor
            observed RV's data

        Returns
        -------
        tf.Tensor
            log prob tensor
        """
        encoding = self.encode_data(x)
        return self.parameters_log_prob_conditioned_to_encoding(
            parameters=parameters,
            encoding=encoding
        )

    def compile(
        self,
        train_method: Literal[
            "reverse_KL",
            "unregularized_ELBO",
            "forward_KL"
        ],
        n_theta_draws_per_x: int,
        **kwargs
    ) -> None:
        """Wrapper for Keras compilation, additionally
        specifying hyper-parameters

        Parameters
        ----------
        train_method : Literal[
            "reverse_KL",
            "unregularized_ELBO",
            "forward_KL"
        ]
            defines which loss to use during training
        n_theta_draws_per_x : int
            for Monte Carlo estimation of the ELBO
            (not used if train_method = "forwardKL")

        Raises
        ------
        NotImplementedError
            train_method not in [
                "reverse_KL",
                "unregularized_ELBO",
                "forward_KL"
            ]
        """
        if train_method not in [
            "reverse_KL",
            "unregularized_ELBO",
            "forward_KL"
        ]:
            raise NotImplementedError(
                f"unrecognized train method {train_method}"
            )
        self.train_method = train_method
        self.n_theta_draws_per_x = n_theta_draws_per_x

        super().compile(**kwargs)

    def train_step(
        self,
        train_data: Tuple[Dict[str, tf.Tensor]]
    ) -> Dict[str, tf.Tensor]:
        """Performs one train step over the fed data
        behavior depends on compiled loss

        Parameters
        ----------
        train_data : Tuple[Dict[str, tf.Tensor]]
            ({rv: data},) conteining batch of training data

        Returns
        -------
        Dict[str, tf.Tensor]
            dict {loss: value}
        """
        data = train_data[0]
        x = data[self.observed_rv]
        if self.train_method == "forward_KL":
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(
                    - self.parameters_log_prob_conditioned_to_data(
                        parameters={
                            rv: data[rv]
                            for rv in self.latent_rvs
                        },
                        x=x
                    )
                )
        elif self.train_method in [
            "reverse_KL",
            "unregularized_ELBO"
        ]:
            repeated_x = tf.repeat(
                x,
                repeats=(self.n_theta_draws_per_x,),
                axis=0
            )
            with tf.GradientTape() as tape:
                encoding = self.encode_data(repeated_x)
                parameters_sample = (
                    self
                    .sample_parameters_conditioned_to_encoding(
                        encoding
                    )
                )

                p = self.generative_hbm.log_prob(
                    **parameters_sample,
                    **{
                        self.observed_rv: repeated_x
                    }
                )

                if self.train_method == "unregularized_ELBO":
                    loss = tf.reduce_mean(-p)
                else:

                    q = (
                        self
                        .parameters_log_prob_conditioned_to_encoding(
                            parameters=parameters_sample,
                            encoding=encoding
                        )
                    )

                    loss = tf.reduce_mean(q - p)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(
            zip(
                (
                    tf.where(
                        tf.math.is_nan(
                            grad
                        ),
                        tf.zeros_like(grad),
                        grad
                    )
                    if grad is not None
                    else None
                    for grad in gradients
                ),
                trainable_vars
            )
        )

        return {self.train_method: loss}


class CascadingFlows(tf.keras.Model):
    """see Ambrogioni et al., 2021"""

    def __init__(
        self,
        generative_hbm: tfd.JointDistributionNamed,
        observed_rvs: List[str],
        link_functions: Dict[str, tfb.Bijector],
        observed_rv_reshapers: Dict[str, tfb.Bijector],
        auxiliary_variables_size: int,
        rff_kwargs: Dict,
        nf_kwargs: Dict,
        amortized: bool,
        auxiliary_target_type: Literal["identity", "MF"],
        **kwargs
    ):
        """Cascading flows architecture for HBM inference

        Parameters
        ----------
        generative_hbm : tfd.JointDistributionNamed
            HBM to perform inference upon
        observed_rvs : List[str]
            [rv] list of observed RVs
        link_functions : Dict[str, tfb.Bijector]
            {rv: link_function}
        observed_rv_reshapers : Dict[str, tfb.Bijector]
            {rv: reshaper}
        auxiliary_variables_size : int
            used to instantiate the auxiliary graph
        rff_kwargs : Dict
            used for observed RV embedders
        nf_kwargs : Dict
            build_trainable_highway_flow kwargs
        amortized : bool
            is the architecture to be amortized
        auxiliary_target_type : Literal["identity", "MF"]
            describes the type of the auxiliary target distribution
            r for the augmented ELBO
        """
        super().__init__(**kwargs)

        self.generative_hbm = generative_hbm
        self.observed_rvs = observed_rvs
        self.link_functions = link_functions
        self.reshapers = observed_rv_reshapers

        self.auxiliary_variables_size = auxiliary_variables_size
        self.amortized = amortized
        self.auxiliary_target_type = auxiliary_target_type

        self.analyse_generative_hbm_graph()
        self.build_architecture(
            rff_kwargs=rff_kwargs,
            nf_kwargs=nf_kwargs
        )

    def analyse_generative_hbm_graph(self) -> None:
        """Analyses dependencies in HBM's graph
        """
        graph = self.generative_hbm.resolve_graph()

        self.prior_rv_order = get_inference_order(
            graph=graph
        )

        self.parents = defaultdict(lambda: list())
        self.children = defaultdict(lambda: list())

        for child, parents in graph:
            for parent in parents:
                self.parents[child].append(parent)
                self.children[parent].append(child)

        self.inverse_rv_order = get_inference_order(
            graph=tuple(
                (rv, tuple(self.children[rv]))
                for rv in self.prior_rv_order
            )
        )

    def build_architecture(
        self,
        rff_kwargs: Dict,
        nf_kwargs: Dict
    ) -> None:
        """Builds whole architecture: auxiliary graph,
        normalizing flows, and observed RV embedders

        Parameters
        ----------
        rff_kwargs : Dict
            used for observed RV embedders
        nf_kwargs : Dict
            build_trainable_highway_flow kwargs
        """
        # ? First deal with the auxiliary model:
        self.auxiliary_coupling_weights = {}
        if self.amortized:
            self.amortizing_bijectors = {}
        auxiliary_model = {}

        for rv in self.inverse_rv_order:
            children = self.children[rv]
            if (
                rv in self.observed_rvs
                and self.amortized
            ):
                self.amortizing_bijectors[rv] = ConditionalAffine(
                    scale_type="none",
                    rff_kwargs=rff_kwargs,
                    event_size=self.auxiliary_variables_size
                )
            if len(children) == 0:
                base_dist = tfd.Independent(
                    tfd.Normal(
                        loc=tf.zeros((self.auxiliary_variables_size,)),
                        scale=1
                    ),
                    reinterpreted_batch_ndims=1
                )
                auxiliary_model[rv] = (
                    tfd.TransformedDistribution(
                        base_dist,
                        bijector=self.amortizing_bijectors[rv]
                    )
                    if rv in self.observed_rvs
                    and self.amortized
                    else
                    base_dist
                )
            else:
                self.auxiliary_coupling_weights[rv] = (
                    tfp.util
                    .TransformedVariable(
                        tf.ones((len(children) + 1,))
                        /
                        (len(children) + 1),
                        bijector=tfb.SoftmaxCentered()
                    )
                )
                auxiliary_model[rv] = eval(
                    f"""lambda {', '.join(children)}: (
                        tfd.TransformedDistribution(
                            tfd.Independent(
                                tfd.Normal(
                                    loc=tf.squeeze(
                                        tf.stack(
                                            [{', '.join(children)}],
                                            axis=-1
                                        )
                                        @
                                        tf.expand_dims(
                                            self.auxiliary_coupling_weights[rv][:-1],
                                            axis=-1
                                        ),
                                        axis=-1
                                    ),
                                    scale=self.auxiliary_coupling_weights[rv][-1]
                                ),
                                reinterpreted_batch_ndims=1
                            ),
                            bijector=self.amortizing_bijectors[rv]
                        )
                        if rv in self.observed_rvs
                        and self.amortized
                        else
                        tfd.Independent(
                            tfd.Normal(
                                loc=tf.squeeze(
                                    tf.stack(
                                        [{', '.join(children)}],
                                        axis=-1
                                    )
                                    @
                                    tf.expand_dims(
                                        self.auxiliary_coupling_weights[rv][:-1],
                                        axis=-1
                                    ),
                                    axis=-1
                                ),
                                scale=self.auxiliary_coupling_weights[rv][-1]
                            ),
                            reinterpreted_batch_ndims=1
                        )
                    )""",
                    {
                        "tfd": tfd,
                        "tf": tf,
                        "self": self,
                        "rv": rv
                    }
                )
        self.auxiliary_model = auxiliary_model

        # ? We then construct the HighwayFlow bijectors
        self.hflows = {}
        self.prior_model = {}
        for rv in self.prior_rv_order:
            if rv in self.observed_rvs:
                continue
            shape = (
                self
                .generative_hbm
                .event_shape
                [rv]
            )
            constrained_shape = (
                self
                .link_functions[rv]
                .inverse_event_shape(
                    shape
                )
            )
            event_size = tf.reduce_prod(
                constrained_shape
            )
            self.reshapers[rv] = tfb.Reshape(
                event_shape_in=(event_size,),
                event_shape_out=constrained_shape
            )

            self.prior_model[rv] = self.generative_hbm.model[rv]

            self.hflows[rv] = (
                tfp.experimental.bijectors
                .build_trainable_highway_flow(
                    width=event_size + self.auxiliary_variables_size,
                    activation_fn=tf.nn.softplus,
                    gate_first_n=event_size,
                    **nf_kwargs
                )
            )

        self.constrainers = {}
        for rv in self.prior_rv_order:
            self.constrainers[rv] = tfb.Chain(
                [
                    self.link_functions[rv],
                    self.reshapers[rv]
                ]
            )

        # ? Finally, we build the auxiliary variational density
        self.auxiliary_target_model = {}
        if self.auxiliary_target_type == "identity":
            for rv in self.prior_rv_order:
                if rv in self.observed_rvs:
                    continue
                self.auxiliary_target_model[rv] = auxiliary_model[rv]
        elif self.auxiliary_target_type == "MF":
            self.auxiliary_MF_locs = {}
            self.auxiliary_MF_scales = {}
            for rv in self.prior_rv_order:
                if rv in self.observed_rvs:
                    continue
                self.auxiliary_MF_locs[rv] = tf.Variable(
                    tf.zeros((self.auxiliary_variables_size,))
                )
                fill_triangular = tfb.FillTriangular()
                self.auxiliary_MF_scales[rv] = tfp.util.TransformedVariable(
                    tf.eye(self.auxiliary_variables_size),
                    bijector=fill_triangular
                )

                self.auxiliary_target_model[rv] = tfd.TransformedDistribution(
                    tfd.MultivariateNormalDiag(
                        loc=tf.zeros((self.auxiliary_variables_size,)),
                        scale_diag=tf.ones((self.auxiliary_variables_size,))
                    ),
                    tfb.Chain([
                        tfb.Shift(self.auxiliary_MF_locs[rv]),
                        tfb.ScaleMatvecTriL(self.auxiliary_MF_scales[rv])
                    ])
                )

    def sample_parameters_conditioned_to_data(
        self,
        data: Dict[str, tf.Tensor],
        return_internals: bool = False
    ) -> Tuple[Dict[str, tf.Tensor], ...]:
        """samples from auxiliary graph, then from prior,
        cascading NF transforms on the prior sample values

        Parameters
        ----------
        data : Dict[str, tf.Tensor]
            {observed_rv: value}
        return_internals : bool, optional
            return intermediate results useful for
            other methods, by default False

        Returns
        -------
        Tuple[Dict[str, tf.Tensor], ...]
            {latent_rv: value}
        """
        batch_size = list(data.values())[0].shape[0]

        auxiliary_values = {}
        for rv in self.inverse_rv_order:
            if (
                self.amortized
                and
                rv in self.observed_rvs
            ):
                conditional_dict = dict(
                    bijector_kwargs=dict(
                        conditional_input=(
                            self.constrainers[rv]
                            .inverse(
                                data[rv]
                            )
                        )
                    )
                )
            else:
                conditional_dict = dict()

            if issubclass(
                type(self.auxiliary_model[rv]),
                tfd.Distribution
            ):
                auxiliary_values[rv] = (
                    self.auxiliary_model[rv]
                    .sample(
                        (batch_size or 1,),
                        **conditional_dict
                    )
                )
            else:
                auxiliary_values[rv] = (
                    self.auxiliary_model[rv](
                        *[
                            auxiliary_values[child]
                            for child in self.children[rv]
                        ]
                    )
                    .sample(
                        tuple(),
                        **conditional_dict
                    )
                )

        augmented_prior_values = {}
        augmented_posterior_values = {}
        sample = {}
        for rv in self.prior_rv_order:
            if rv in self.observed_rvs:
                continue
            prior_value = self.constrainers[rv].inverse(
                self.prior_model[rv].sample((batch_size or 1,))
                if issubclass(type(self.prior_model[rv]), tfd.Distribution)
                else
                self.prior_model[rv](
                    *[
                        {
                            **data,
                            **sample
                        }[parent]
                        for parent in self.parents[rv]
                    ]
                ).sample()
            )

            augmented_prior_values[rv] = tf.concat(
                [
                    prior_value,
                    auxiliary_values[rv]
                ],
                axis=-1
            )
            augmented_posterior_values[rv] = (
                self.hflows[rv]
                .forward(augmented_prior_values[rv])
            )
            sample[rv] = self.constrainers[rv].forward(
                augmented_posterior_values[rv]
                [..., :-self.auxiliary_variables_size]
            )

        if return_internals:
            return (
                sample,
                augmented_posterior_values,
                augmented_prior_values,
                auxiliary_values
            )
        else:
            return (sample,)

    def joint_log_prob_conditioned_to_data(
        self,
        data: Dict[str, tf.Tensor],
        augmented_posterior_values: Dict[str, tf.Tensor],
        auxiliary_values: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """Needs auxiliary values to compute latent RVs log prob conditional
        on observed RV's value

        Parameters
        ----------
        data : Dict[str, tf.Tensor]
            {rv: value}
        augmented_posterior_values : Dict[str, tf.Tensor]
            {rv: augmented_value}
        auxiliary_values : Dict[str, tf.Tensor]
            {rv: auxiliary_value}

        Returns
        -------
        tf.Tensor
            log prob tensor
        """

        batch_size = list(data.values())[0].shape[0]

        log_prob = 0.
        for rv in self.prior_rv_order:
            if rv in self.observed_rvs:
                continue
            prior_dist = tfd.TransformedDistribution(
                tfd.BatchBroadcast(
                    self.prior_model[rv],
                    to_shape=(batch_size,)
                )
                if issubclass(type(self.prior_model[rv]), tfd.Distribution)
                else
                self.prior_model[rv](
                    *[
                        data[parent]
                        for parent in self.parents[rv]
                    ]
                ),
                bijector=tfb.Invert(
                    self.constrainers[rv]
                )
            )
            auxiliary_dist = (
                self.auxiliary_model[rv]
                if issubclass(type(self.auxiliary_model[rv]), tfd.Distribution)
                else
                self.auxiliary_model[rv](
                    *[
                        auxiliary_values[child]
                        for child in self.children[rv]
                    ]
                )
            )

            augmented_prior_dist = tfd.Blockwise(
                [
                    prior_dist,
                    auxiliary_dist
                ]
            )
            augmented_posterior_dist = tfd.TransformedDistribution(
                augmented_prior_dist,
                bijector=self.hflows[rv]
            )

            log_prob += augmented_posterior_dist.log_prob(
                augmented_posterior_values[rv]
            )

        return log_prob

    def MF_log_prob(
        self,
        augmented_posterior_values: Dict[str, tf.Tensor],
        auxiliary_values: Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        """target auxiliary density log prob for
        augmented ELBO

        Parameters
        ----------
        augmented_posterior_values : Dict[str, tf.Tensor]
            {rv: augmented_value}
        auxiliary_values : Dict[str, tf.Tensor]
            {rv: auxiliary_value}

        Returns
        -------
        tf.Tensor
            log prob tensor
        """
        batch_size = list(
            augmented_posterior_values
            .values()
        )[0].shape[0]

        log_prob = 0.
        for rv in self.prior_rv_order:
            if rv in self.observed_rvs:
                continue
            if self.auxiliary_target_type == "identity":
                auxiliary_target_dist = (
                    tfd.Sample(
                        self.auxiliary_target_model[rv],
                        sample_shape=(batch_size,)
                    )
                    if issubclass(
                        type(self.auxiliary_target_model[rv]),
                        tfd.Distribution
                    )
                    else
                    self.auxiliary_target_model[rv](
                        *[
                            auxiliary_values[child]
                            for child in self.children[rv]
                        ]
                    )
                )
                log_prob += (
                    auxiliary_target_dist
                    .log_prob(
                        augmented_posterior_values[rv]
                        [..., -self.auxiliary_variables_size:]
                    )
                )
            elif self.auxiliary_target_type == "MF":
                log_prob += (
                    self.auxiliary_target_model[rv]
                    .log_prob(
                        augmented_posterior_values[rv]
                        [..., -self.auxiliary_variables_size:]
                    )
                )
        return log_prob

    def compile(
        self,
        train_method: Literal[
            "reverse_KL",
            "unregularized_ELBO"
        ],
        n_theta_draws_per_x: int,
        **kwargs
    ) -> None:
        """Wrapper for Keras compilation, additionally
        specifying hyper-parameters

        Parameters
        ----------
        train_method : Literal[
            "reverse_KL",
            "unregularized_ELBO"
        ]
            defines which loss to use during training
        n_theta_draws_per_x : int
            for Monte Carlo estimation of the ELBO
            (not used if train_method = "forwardKL")

        Raises
        ------
        NotImplementedError
            train_method not in [
                "reverse_KL",
                "unregularized_ELBO"
            ]
        """
        if train_method not in [
            "reverse_KL",
            "unregularized_ELBO"
        ]:
            raise NotImplementedError(
                f"unrecognized train method {train_method}"
            )
        self.train_method = train_method
        self.n_theta_draws_per_x = n_theta_draws_per_x

        super().compile(**kwargs)

    def train_step(
        self,
        train_data: Tuple[Dict[str, tf.Tensor]]
    ) -> Dict[str, tf.Tensor]:
        """Performs a train step on training data (behavior depends
        on compiled train method)

        Parameters
        ----------
        train_data : Tuple[Dict[str, tf.Tensor]]
            tuple containing the {rv: value} dict
            corresponding to the train data batch

        Returns
        -------
        Dict[str, tf.Tensor]
            {train_method: loss_value}
        """
        data = train_data[0]
        if self.train_method in [
            "reverse_KL",
            "unregularized_ELBO"
        ]:
            repeated_rvs = {
                rv: tf.repeat(
                    value,
                    repeats=(self.n_theta_draws_per_x,),
                    axis=0
                )
                for rv, value in data.items()
            }
            with tf.GradientTape() as tape:
                (
                    parameters_sample,
                    augmented_posterior_values,
                    _,
                    auxiliary_values
                ) = self.sample_parameters_conditioned_to_data(
                    data=repeated_rvs,
                    return_internals=True
                )

                p = self.generative_hbm.log_prob(
                    **parameters_sample,
                    **{
                        observed_rv: repeated_rvs[observed_rv]
                        for observed_rv in self.observed_rvs
                    }
                )

                if self.train_method == "unregularized_ELBO":
                    loss = tf.reduce_mean(-p)
                else:
                    r = self.MF_log_prob(
                        augmented_posterior_values=augmented_posterior_values,
                        auxiliary_values=auxiliary_values,
                    )

                    q = (
                        self
                        .joint_log_prob_conditioned_to_data(
                            data={
                                **parameters_sample,
                                **{
                                    observed_rv: repeated_rvs[observed_rv]
                                    for observed_rv in self.observed_rvs
                                }
                            },
                            augmented_posterior_values=(
                                augmented_posterior_values
                            ),
                            auxiliary_values=auxiliary_values
                        )
                    )

                    loss = tf.reduce_mean(q - p - r)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(
            zip(
                (
                    tf.where(
                        tf.math.is_nan(
                            grad
                        ),
                        tf.zeros_like(grad),
                        grad
                    )
                    if grad is not None
                    else None
                    for grad in gradients
                ),
                trainable_vars
            )
        )

        return {self.train_method: loss}


class ADAVFamily(tf.keras.Model):

    def __init__(
        self,
        generative_hbm: Union[
            tfd.JointDistributionNamed,
            tfd.JointDistributionCoroutine
        ],
        set_transforer_kwargs: Dict,
        conditional_nf_chain_kwargs: Dict,
        hierarchies: Dict,
        link_functions: Dict,
        **kwargs
    ):
        """Automatically constructs the amortized dual variational
        family from the input generative_hbm

        Parameters
        ----------
        generative_hbm : Union[
            tfd.JointDistributionNamed,
            tfd.JointDistributionCoroutine
        ]
            generative Hierarchical Bayesian Model
            on which to perform inference
        set_transforer_kwargs : Dict
            for hierarchical encoder
        conditional_nf_chain_kwargs : Dict
            for conditional density estimators
        hierarchies : Dict
            dict of {key: hierarchy} for all keys
            in the generative_hbm
        link_functions : Dict
            dict of {key: tfb.Bijector} for all keys
            in the generative_hbm
        """
        super().__init__(**kwargs)

        self.generative_hbm = generative_hbm
        self.hierarchies = hierarchies
        self.link_functions = link_functions

        self.analyse_generative_hbm_graph()
        self.build_architecture(
            set_transforer_kwargs=set_transforer_kwargs,
            conditional_nf_chain_kwargs=conditional_nf_chain_kwargs
        )

    def analyse_generative_hbm_graph(self) -> None:
        """Creates internals related to hierarchies
        """
        self.max_hierarchy = max(
            self
            .hierarchies
            .values()
        )
        self.keys_per_hierarchy = {
            h: [
                key
                for key, value in (
                    self
                    .hierarchies
                    .items()
                )
                if value == h
            ]
            for h in range(self.max_hierarchy + 1)
        }
        self.input_key = self.keys_per_hierarchy.pop(0)[0]

    def build_architecture(
        self,
        set_transforer_kwargs: Dict,
        conditional_nf_chain_kwargs: Dict
    ) -> None:
        """Creates parametric hierarchical encoder
        and conditional density estimators

        Parameters
        ----------
        set_transforer_kwargs : Dict
            for hierarchical encoder
        conditional_nf_chain_kwargs : Dict
            for conditional density estimators
        """

        self.set_transformers = {}
        self.conditional_density_estimators = {}
        self.exp_conditional_affine_density_estimators = {}

        for h in range(1, self.max_hierarchy + 1):
            self.set_transformers[h] = SetTransformer(
                **set_transforer_kwargs,
                attention_axes=(-3,)
            )

            embedding_size = (
                self
                .set_transformers[h]
                .embedding_size
            )

            for key in self.keys_per_hierarchy[h]:
                shape = (
                    self
                    .generative_hbm
                    .event_shape
                    [key]
                    if (
                        type(self.generative_hbm)
                        ==
                        tfd.JointDistributionNamed
                    )
                    else
                    self
                    .generative_hbm
                    .event_shape
                    ._asdict()
                    [key]
                )

                constrained_shape = (
                    self
                    .link_functions[key]
                    .inverse_event_shape(
                        shape
                    )
                )

                batch_shape = constrained_shape[:self.max_hierarchy - h]
                event_size = tf.reduce_prod(
                    constrained_shape[self.max_hierarchy - h:]
                )
                latent_shape = (
                    batch_shape
                    +
                    (event_size,)
                )

                latent_distribution = tfd.Independent(
                    tfd.Normal(
                        loc=tf.zeros(latent_shape),
                        scale=1.0
                    ),
                    reinterpreted_batch_ndims=self.max_hierarchy - h + 1
                )

                reshaper = tfb.Reshape(
                    event_shape_out=constrained_shape,
                    event_shape_in=latent_shape
                )

                nf = ConditionalNFChain(
                    event_size=event_size.numpy(),
                    conditional_event_size=embedding_size,
                    name=f"nf_{key}",
                    **conditional_nf_chain_kwargs
                )

                self.conditional_density_estimators[key] = (
                    tfd.TransformedDistribution(
                        distribution=latent_distribution,
                        bijector=tfb.Chain(
                            bijectors=[
                                self.link_functions[key],
                                reshaper,
                                nf
                            ]
                        )
                    )
                )

                # ! EXPERIMENTAL - needs to be refactored
                # assumes that conditional_nf_chain.bijectors[-1]
                # is a ConditionalAffine bijector
                self.exp_conditional_affine_density_estimators[key] = (
                    tfd.TransformedDistribution(
                        distribution=latent_distribution,
                        bijector=tfb.Chain(
                            bijectors=[
                                self.link_functions[key],
                                reshaper,
                                nf.bijectors[-1]
                            ]
                        )
                    )
                )

    def encode_data(
        self,
        x: tf.Tensor
    ) -> Dict:
        """Encodes input data x
        via stacked SetTransformers

        Parameters
        ----------
        x : tf.Tensor
            shape:
                (batch_size,)
                + (
                    Card(P_p),
                    ...,
                    Card(P_0)
                )
                + Shape(x)

        Returns
        -------
        Dict
            encodings from various hierarchies
            shapes: {
                hierarchy:
                    (batch_size,)
                    + (
                        Card(P_p),
                        ...,
                        Card(P_hierarchy)
                    )
                    + (e_hierarchy,)
            }
        """
        z = (
            self
            .link_functions[self.input_key]
            .inverse(x)
        )
        encodings = {}
        for h in range(1, self.max_hierarchy + 1):
            z = self.set_transformers[h](z)
            z = tf.squeeze(z, axis=-2)

            encodings[h] = z

        return encodings

    def sample_parameters_conditioned_to_encodings(
        self,
        encodings: Dict
    ) -> Dict:
        """sample a single point from conditional density estimators

        Parameters
        ----------
        encodings : Dict
            encodings from various hierarchies
            shapes: {
                hierarchy:
                    (batch_size,)
                    + (
                        Card(P_p),
                        ...,
                        Card(P_hierarchy)
                    )
                    + (e_hierarchy,)
            }

        Returns
        -------
        Dict
            sample
            shapes: {
                key:
                    (batch_size,)
                    + (
                        Card(P_p),
                        ...,
                        Card(P_hierarchy)
                    )
                    + Shape(key)
            }
        """
        batch_size = encodings[1].shape[0]
        sample = {}
        for h in range(1, self.max_hierarchy + 1):
            for key in self.keys_per_hierarchy[h]:
                sample[key] = (
                    self
                    .conditional_density_estimators[key]
                    .sample(
                        (batch_size,),
                        bijector_kwargs={
                            f"nf_{key}": dict(
                                conditional_input=encodings[h]
                            )
                        }
                    )
                )

        return sample

    def sample_parameters_conditioned_to_data(
        self,
        x: tf.Tensor
    ) -> Dict:
        """Wrapper for encode_data
        followed by sample_parameters_conditioned_to_encodings
        """
        encodings = self.encode_data(x)

        return self.sample_parameters_conditioned_to_encodings(encodings)

    def parameters_log_prob_conditioned_to_encodings(
        self,
        parameters: Dict,
        encodings: Dict
    ) -> tf.Tensor:
        """Calculate posterior log prob for the parameters

        Parameters
        ----------
        parameters : Dict
            shapes: {
                key:
                    (batch_size,)
                    + (
                        Card(P_p),
                        ...,
                        Card(P_hierarchy)
                    )
                    + Shape(key)
            }
        encodings : Dict
            encodings from various hierarchies
            shapes: {
                hierarchy:
                    (batch_size,)
                    + (
                        Card(P_p),
                        ...,
                        Card(P_hierarchy)
                    )
                    + (e_hierarchy,)
            }

        Returns
        -------
        tf.Tensor
            shape: (batch_size,)
        """
        log_prob = 0
        for h in range(1, self.max_hierarchy + 1):
            for key in self.keys_per_hierarchy[h]:
                log_prob += (
                    self
                    .conditional_density_estimators[key]
                    .log_prob(
                        parameters[key],
                        bijector_kwargs={
                            f"nf_{key}": dict(
                                conditional_input=encodings[h]
                            )
                        }
                    )
                )

        return log_prob

    def parameters_log_prob_conditioned_to_data(
        self,
        parameters: Dict,
        x: tf.Tensor
    ) -> tf.Tensor:
        """Wrapper for encode_data
        followed by parameters_log_prob_conditioned_to_encodings
        """
        encodings = self.encode_data(x)

        return self.parameters_log_prob_conditioned_to_encodings(
            parameters=parameters,
            encodings=encodings
        )

    def exp_affine_MAP_regression_conditioned_to_encodings(
        self,
        encodings: Dict
    ) -> Dict:
        """# ! EXPERIMENTAL - needs to be refactored
        Assumes conditional_nf_chains.bijectors[-1] to be a
        ConditionalAffine bijector, from which we retrieve the shift
        regressor and apply it to the encodings

        Parameters
        ----------
        encodings : Dict
            encodings from various hierarchies
            shapes: {
                hierarchy:
                    (batch_size,)
                    + (
                        Card(P_p),
                        ...,
                        Card(P_hierarchy)
                    )
                    + (e_hierarchy,)
            }

        Returns
        -------
        Dict
            map values
            shapes: {
                key:
                    (batch_size,)
                    + (
                        Card(P_p),
                        ...,
                        Card(P_hierarchy)
                    )
                    + Shape(key)
            }
        """
        map_values = {}
        for h in range(1, self.max_hierarchy + 1):
            for key in self.keys_per_hierarchy[h]:
                map_values[key] = tfb.Chain(
                    bijectors=(
                        self
                        .conditional_density_estimators[key]
                        .bijector
                        .bijectors[:-1]
                    )
                )(
                    self
                    .conditional_density_estimators[key]
                    .bijector
                    .bijectors[-1]
                    .bijectors[-1]
                    .shift(
                        encodings[h]
                    )
                )
        return map_values

    def exp_affine_sample_parameters_conditioned_to_encodings(
        self,
        encodings: Dict
    ) -> Dict:
        """# ! EXPERIMENTAL - needs to be refactored
        sample a single point from conditional affine density estimators

        Parameters
        ----------
        encodings : Dict
            encodings from various hierarchies
            shapes: {
                hierarchy:
                    (batch_size,)
                    + (
                        Card(P_p),
                        ...,
                        Card(P_hierarchy)
                    )
                    + (e_hierarchy,)
            }

        Returns
        -------
        Dict
            sample
            shapes: {
                key:
                    (batch_size,)
                    + (
                        Card(P_p),
                        ...,
                        Card(P_hierarchy)
                    )
                    + Shape(key)
            }
        """
        batch_size = encodings[1].shape[0]
        sample = {}
        for h in range(1, self.max_hierarchy + 1):
            for key in self.keys_per_hierarchy[h]:
                sample[key] = (
                    self
                    .exp_conditional_affine_density_estimators[key]
                    .sample(
                        (batch_size,),
                        bijector_kwargs={
                            (
                                self
                                .exp_conditional_affine_density_estimators[key]
                                .bijector
                                .bijectors[-1]
                                .name
                            ): dict(
                                conditional_input=encodings[h]
                            )
                        }
                    )
                )

        return sample

    def compile(
        self,
        train_method: Literal[
            "forward_KL",
            "reverse_KL",
            "unregularized_ELBO",
            "exp_MAP_regression",
            "exp_affine_unregularized_ELBO"
        ],
        n_theta_draws_per_x: int,
        **kwargs
    ) -> None:
        """Wrapper for Keras compilation, additionally
        specifying hyper-parameters

        Parameters
        ----------
        train_method : Literal[
            "forward_KL",
            "reverse_KL",
            "unregularized_ELBO",
            "exp_MAP_regression",
            "exp_affine_unregularized_ELBO"
        ]
            defines which loss to use during training
        n_theta_draws_per_x : int
            for Monte Carlo estimation of the ELBO
            (not used if train_method = "forwardKL")

        Raises
        ------
        NotImplementedError
            train_method not in [
                "forward_KL",
                "reverse_KL",
                "unregularized_ELBO",
                "exp_MAP_regression",
                "exp_affine_unregularized_ELBO"
            ]
        """
        if train_method not in [
            "forward_KL",
            "reverse_KL",
            "unregularized_ELBO",
            "exp_MAP_regression",
            "exp_affine_unregularized_ELBO"
        ]:
            raise NotImplementedError(
                f"unrecognized train method {train_method}"
            )
        self.train_method = train_method
        self.n_theta_draws_per_x = n_theta_draws_per_x

        super().compile(**kwargs)

    def train_step(
        self,
        data: Tuple[Dict]
    ) -> Dict:
        """Keras train step

        Parameters
        ----------
        data : Tuple[Dict]
            data from the generative_hbm
            various keys will be used depending
            on the train_method

        Returns
        -------
        Dict
            {loss_type: value} depending on
            the train method
        """
        x = data[0][self.input_key]
        if self.train_method == "forward_KL":
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(
                    - self.parameters_log_prob_conditioned_to_data(
                        parameters=data[0],
                        x=x
                    )
                )
        elif self.train_method in [
            "reverse_KL",
            "unregularized_ELBO"
        ]:
            repeated_x = tf.repeat(
                x,
                repeats=(self.n_theta_draws_per_x,),
                axis=0
            )
            with tf.GradientTape() as tape:
                encodings = self.encode_data(
                    x=repeated_x
                )
                parameters_sample = (
                    self
                    .sample_parameters_conditioned_to_encodings(
                        encodings=encodings
                    )
                )
                p = self.generative_hbm.log_prob(
                    **parameters_sample,
                    **{
                        self.input_key: repeated_x
                    }
                )
                if self.train_method == "unregularized_ELBO":
                    loss = tf.reduce_mean(-p)
                else:
                    q = (
                        self
                        .parameters_log_prob_conditioned_to_encodings(
                            parameters=parameters_sample,
                            encodings=encodings
                        )
                    )
                    loss = tf.reduce_mean(q - p)
        elif self.train_method == "exp_MAP_regression":
            with tf.GradientTape() as tape:
                encodings = self.encode_data(
                    x=x
                )
                map_values = (
                    self
                    .exp_affine_MAP_regression_conditioned_to_encodings(
                        encodings=encodings
                    )
                )
                p = self.generative_hbm.log_prob(
                    **map_values,
                    **{
                        self.input_key: x
                    }
                )
                loss = tf.reduce_mean(-p)
        elif self.train_method == "exp_affine_unregularized_ELBO":
            repeated_x = tf.repeat(
                x,
                repeats=(self.n_theta_draws_per_x,),
                axis=0
            )
            with tf.GradientTape() as tape:
                encodings = self.encode_data(
                    x=repeated_x
                )
                parameters_sample = (
                    self
                    .exp_affine_sample_parameters_conditioned_to_encodings(
                        encodings=encodings
                    )
                )
                p = self.generative_hbm.log_prob(
                    **parameters_sample,
                    **{
                        self.input_key: repeated_x
                    }
                )
                loss = tf.reduce_mean(-p)

        trainable_vars = self.trainable_variables
        for estimator in self.conditional_density_estimators.values():
            trainable_vars += estimator.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(
            zip(
                (
                    tf.where(
                        tf.math.is_nan(
                            grad
                        ),
                        tf.zeros_like(grad),
                        grad
                    )
                    if grad is not None
                    else None
                    for grad in gradients
                ),
                trainable_vars
            )
        )
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {self.train_method: loss}

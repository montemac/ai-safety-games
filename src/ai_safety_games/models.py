# TODO: implement a DecisionTransformer by extending the normal hooked
# transformer class from TransformerLens, with the following changes:
# D1. self.embed is overwritten after the parent __init__ call to be
#    Identity.
# D2. self.pos_embed is overwritten after the parent __init__ call to
#    embed the timestep, not the position, i.e. the timestep is the same
#    for each of the RSA inputs for that timestep.
# D3. self.unembed is overwritten after the parent __init__ call to
#    unembeded just the next action prediction (i.e. don't predict the
#    reward to go and state, since these weren't reported as helping in
#    the original decision transformer paper).  Maybe this should be
#    optional?
# D4. A call to self.setup() is made after these module updates to
#    rebuild the hook point dict.
# 5. Overrides self.loss_fn() to calculate loss from the RSA inputs and
#    the logits (needed because we can predict the action at each
#    timestep, we don't need the index-off-by-one needed when predicting
#    next tokens.)
# D6. Overrides various token-related methods to make them do nothing
# D7. Define a new config class, and map this into the
#    HookedTransformerConfig class before calling the parent __init__.
# D8. Override the forward() method to prevent key-value caching and
#    remove other irrelevant args (e.g. BOS), and do RSA embedding.
#    This is needed because the HookedTransformer forward() method can't
#    take a tuple of inputs, so we need to RSA embed first.
# D9. Make forward call only ever evaluate logits, because we don't have
#    tokens in the forward call to use for loss calcs.  Loss calcs must
#    be done later using the logits.
#
# NOTE: inheriting from HookedTransformer is a bit hacky, but seemed
# like the least broken thing overall!  This is despite the fact that
# the forward call signature is different.  I think the code reuse makes
# up for this, but this is a weakly held opinion.

from typing import List, Union, Optional, Tuple, Dict
from dataclasses import dataclass, asdict, field, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from einops import rearrange, einsum, repeat
import numpy as np

from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    utils,
    utilities,
)
from transformer_lens.components import Float, Int, torch


class RSAEmbed(nn.Module):
    """Embeds the RSA inputs into the model's latent space."""

    def __init__(self, d_state: int, d_action: int, d_model: int):
        """Initialize the RSA embedding module."""
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action
        self.d_model = d_model
        self.W_R: Float[torch.Tensor, "d_model"] = nn.Parameter(
            torch.empty(d_model)
        )
        self.W_S: Float[torch.Tensor, "d_state d_model"] = nn.Parameter(
            torch.empty(d_state, d_model)
        )
        self.W_A: Float[torch.Tensor, "d_action d_model"] = nn.Parameter(
            torch.empty(d_action, d_model)
        )

    def forward(
        self,
        rtgs=Float[torch.Tensor, "batch timestep_r"],
        states=Float[torch.Tensor, "batch timestep_s d_state"],
        actions=Int[torch.Tensor, "batch timestep_a"],
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """Embed the RSA inputs into the model's latent space. Args `rtgs`
        `states` and `actions` have second dimension of timestep, and
        are emedded into the models latent space, then interleaved so
        that resulting tensor's second dimension is pos, with has length
        (timestep_r + timestep_s + timestep_a).  Interleaving is always
        in (RTG, state, action) order. The final timestep can be
        incomplete, i.e. contain either just an RTG, or an RTG and a
        state."""
        # Confirm timestep dims are correct
        assert (
            rtgs.shape[1] >= states.shape[1]
            and states.shape[1] >= actions.shape[1]
        ), "RTG, state, and action timesteps must be in order"
        assert (
            states.shape[1] >= rtgs.shape[1] - 1
            and actions.shape[1] >= rtgs.shape[1] - 1
        ), (
            "State and action timesteps must be "
            + "at least one less than RTG timesteps."
        )
        # Embed the RTGs
        rtgs_embedded = einsum(
            rtgs,
            self.W_R,
            "batch timestep, d_model -> batch timestep d_model",
        )
        # Embed the states
        states_embedded = einsum(
            states,
            self.W_S,
            "batch timestep d_state, d_state d_model"
            + " -> batch timestep d_model",
        )
        # Embed the actions, which are passed as ints, which we use to
        # index into the embedding matrix.
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has
        # shape [a, c, d] B acts as a tensor of indices into the second
        # dimension (so >=0 and <b)
        actions_embedded = self.W_A[actions, :]
        # Interleave the three tensors
        rsa_embedded = torch.zeros(
            (
                rtgs.shape[0],
                rtgs.shape[1] + states.shape[1] + actions.shape[1],
                self.d_model,
            )
        )
        rsa_embedded[:, ::3, :] = rtgs_embedded
        rsa_embedded[:, 1::3, :] = states_embedded
        rsa_embedded[:, 2::3, :] = actions_embedded
        return rsa_embedded
        # return rearrange(
        #     [rtgs_embedded, states_embedded, actions_embedded],
        #     "d_rsa batch timestep d_model"
        #     + " -> batch (timestep d_rsa) d_model",
        # )


class RSAUnembed(nn.Module):
    """Unembeds the model's latent space into the RSA outputs, which are
    next-action predictions given the reward to go and state at each
    timestep. That is, residual stream positions corresponding to the
    state embedding are unembedded to produce logits over the action
    that would follow this state. The resulting logits tensor has second
    dimension of timesteps, not transformer positions."""

    def __init__(self, d_action: int, d_model: int):
        """Initialize the RSA unembedding module."""
        super().__init__()
        self.d_action = d_action
        self.W_U: Float[torch.Tensor, "d_model d_action"] = nn.Parameter(
            torch.empty(d_model, d_action)
        )
        self.b_U: Float[torch.Tensor, "d_action"] = nn.Parameter(
            torch.zeros(d_action)
        )

    def forward(
        self, residual: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_vocab_out"]:
        """Unembed the model's latent space into the RSA outputs, which
        are next-action predictions given the reward to go and state at
        each timestep."""
        return (
            einsum(
                residual[
                    :, 1::3, :
                ],  # Position of the first state embedding, then every third position
                self.W_U,
                "batch timestep d_model, d_model vocab -> batch timestep vocab",
            )
            + self.b_U
        )


class RSAPosEmbed(nn.Module):
    """Embeds the timestep into the model's latent space."""

    def __init__(self, n_timesteps: int, d_model: int):
        """Initialize the RSA positional embedding module."""
        super().__init__()
        self.n_timesteps = n_timesteps
        self.d_model = d_model
        self.W_timestep = nn.Parameter(torch.empty(n_timesteps, d_model))

    def forward(
        self,
        rsa_embeddings=Float[torch.Tensor, "batch pos d_model"],
        past_kv_pos_offset: int = 0,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """Calculate the positional embedding for the RSA input
        timesteps. The same positional embedding will be used at each of
        the three residual stream positions corresponding to each timestep.

        Output shape [pos, d_model] - will be broadcast along batch
        dim"""
        assert past_kv_pos_offset == 0, "past_kv_pos_offset not supported"
        num_pos = rsa_embeddings.size(1)
        timestep_indices = (
            torch.arange(num_pos, device=rsa_embeddings.device) // 3
        )  # [pos]
        pos_embed = self.W_timestep[timestep_indices, :]  # [pos, d_model]
        broadcast_pos_embed = repeat(
            pos_embed,
            "pos d_model -> batch pos d_model",
            batch=rsa_embeddings.shape[0],
        )  # [batch, pos, d_model]
        return broadcast_pos_embed.clone()


@dataclass
class DecisionTransformerConfig:
    """A config for the DecisionTransformer. Mostly just a subset of the
    fields present in HookedTransformerConfig. Processing of these args
    is *not* performed at init time, since this object will be used to
    create a HookedTrnasformerConfig object later.

    See docs for HookedTransformerConfig for description of the fields.

    New args:
        n_timesteps (int): the number of timesteps covered by the
        model's context window.
    """

    # HookedTransformerConfig required fields
    n_layers: int
    d_model: int
    n_ctx: int = field(init=False)
    d_head: int
    # New required fields
    n_timesteps: int
    d_state: int
    d_action: int
    # HookedTransformerConfig default fields
    model_name: str = "custom"
    n_heads: int = -1
    d_mlp: Optional[int] = None
    act_fn: Optional[str] = None
    d_vocab: int = field(init=False)
    eps: float = 1e-5
    use_attn_result: bool = False
    use_attn_scale: bool = True
    use_split_qkv_input: bool = False
    use_local_attn: bool = False
    original_architecture: Optional[str] = None
    from_checkpoint: bool = False
    checkpoint_index: Optional[int] = None
    checkpoint_label_type: Optional[str] = None
    checkpoint_value: Optional[int] = None
    # tokenizer_name: Optional[str] = None
    window_size: Optional[int] = None
    attn_types: Optional[List] = None
    init_mode: str = "gpt2"
    normalization_type: Optional[str] = "LN"
    device: Optional[str] = None
    n_devices: int = 1
    attention_dir: str = "causal"
    attn_only: bool = False
    seed: Optional[int] = None
    initializer_range: float = -1.0
    init_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    # positional_embedding_type: str = "standard"
    final_rms: bool = False
    d_vocab_out: int = field(init=False)
    parallel_attn_mlp: bool = False
    # rotary_dim: Optional[int] = None
    n_params: Optional[int] = None
    use_hook_tokens: bool = False
    gated_mlp: bool = False

    def __post_init__(self):
        """Config post-initialization processing."""
        self.n_ctx = self.n_timesteps * 3
        # This prevents parent class from complaining about missing
        # arguments when we create a HookedTransformerConfig object,
        # pretty hacky.
        self.d_vocab = self.d_action
        self.d_vocab_out = self.d_action


class DecisionTransformer(HookedTransformer):
    """A transformer that takes RSA inputs and predicts the next
    action."""

    def __init__(
        self,
        cfg: Union[DecisionTransformerConfig, Dict],
        move_to_device: bool = True,
    ):
        """Initialize the decision transformer model.

        Args:
            cfg: A DecisionTransformerConfig object, or dict of fields.
            move_to_device: Whether to move the model to the device
                specified in the config.
        """
        if isinstance(cfg, Dict):
            cfg = DecisionTransformerConfig(**cfg)
        self.dt_cfg = cfg
        # Filter out any fields not present in HookedTransformerConfig
        # and create a HookedTransformerConfig object
        cfg_dict = asdict(cfg)
        ht_fields = [field.name for field in fields(HookedTransformerConfig)]
        ht_cfg = HookedTransformerConfig(
            **dict(
                filter(
                    lambda inp: inp[0] in ht_fields,
                    cfg_dict.items(),
                )
            )
        )
        super().__init__(cfg=ht_cfg, move_to_device=move_to_device)

        self.first_device = utilities.devices.get_device_for_block_index(
            0, self.cfg
        )
        self.last_device = utilities.devices.get_device_for_block_index(
            self.cfg.n_layers - 1, self.cfg
        )
        # Create embedding module, move to correct device
        self.rsa_embed = RSAEmbed(
            self.dt_cfg.d_state, self.dt_cfg.d_action, self.dt_cfg.d_model
        ).to(self.first_device)
        # Override a few modules from the parent class, moving to them
        # to the correct devices
        self.embed = nn.Identity().to(self.first_device)
        self.pos_embed = RSAPosEmbed(
            self.dt_cfg.n_timesteps, self.dt_cfg.d_model
        ).to(self.first_device)
        self.unembed = RSAUnembed(
            self.dt_cfg.d_action, self.dt_cfg.d_model
        ).to(self.last_device)

        # Redo module initialization now that we've overridden some
        # (Will re-init all weights, keep that in mind for determinism)
        if self.cfg.init_weights:
            self.init_weights()

        # Redo setup call
        self.setup()

    def forward(
        self,
        rtgs=Float[torch.Tensor, "batch timestep"],
        states=Float[torch.Tensor, "batch timestep d_state"],
        actions=Int[torch.Tensor, "batch timestep"],
        stop_at_layer: Optional[int] = None,
    ) -> Float[
        torch.Tensor, "batch pos d_vocab"
    ]:  # pylint: disable=arguments-differ
        """Forward pass through the model. Calls
        HookedTransformer.forward,
        but modifies some arguments, and does the RSA embedding
        first, passing in the embedded inputs tensors in place of the
        usual tokens tensor."""
        assert (
            rtgs.shape[1] + states.shape[1] + actions.shape[1]
            <= self.cfg.n_ctx
        ), "Input sequence too long!"
        # Embed the RSA inputs
        rsa_embeddings = self.rsa_embed(rtgs, states, actions)
        # Run the HookedTransformer forward pass
        return super().forward(
            rsa_embeddings,
            return_type="logits",
            prepend_bos=False,
            stop_at_layer=stop_at_layer,
            past_kv_cache=None,
        )

    def loss_fn(
        self,
        logits: Float[torch.Tensor, "batch pos d_vocab"],
        actions: Int[torch.Tensor, "batch pos"],
        per_token: bool = False,
    ):
        """Compute the cross-entropy loss between the next-action logits and actual actions."""
        if actions.device != logits.device:
            actions = actions.to(logits.device)
        log_probs = F.log_softmax(logits, dim=-1)
        predicted_log_probs = log_probs.gather(
            dim=-1, index=actions[..., None]
        )[..., 0]
        if per_token:
            return -predicted_log_probs
        else:
            return -predicted_log_probs.mean()

    def sample_next_action(
        self,
        rtgs=Float[torch.Tensor, "batch timestep"],
        states=Float[torch.Tensor, "batch timestep d_state"],
        actions=Int[torch.Tensor, "batch timestep"],
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
    ) -> Int[torch.Tensor, "batch"]:
        """Do a forward pass and sample the next action. Provided
        actions tensor must have data for one less timestep than rtgs
        and states, i.e. the current timestep, for which we want to
        sample an action conditioned on the previous timesteps and
        current rtg/state."""
        # Get logits via a forward pass
        final_logits = self.forward(rtgs, states, actions)[:, -1, :]
        # Sample the next action from the logits
        self.eval()
        sampled_action = utils.sample_logits(
            final_logits,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        ).to(self.first_device)
        self.train()
        return sampled_action

    # Code below here is just overriding methods from HookedTransformer
    # that don't work for DecisionTransformer
    def set_tokenizer(self, tokenizer):
        raise NotImplementedError(
            "Tokenizer not supported for DecisionTransformer"
        )

    def to_tokens(
        self,
        input: Union[str, List[str]],
        prepend_bos: bool = True,
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> Int[torch.Tensor, "batch pos"]:
        raise NotImplementedError(
            "to_tokens not supported for DecisionTransformer"
        )

    def to_string(
        self,
        tokens: Union[
            Int[torch.Tensor, ""],
            Int[torch.Tensor, "batch pos"],
            Int[torch.Tensor, "pos"],
            np.ndarray,
            List[Int[torch.Tensor, "pos"]],
        ],
    ) -> Union[str, List[str]]:
        raise NotImplementedError(
            "to_string not supported for DecisionTransformer"
        )

    def to_str_tokens(
        self,
        input: Union[
            str,
            Int[torch.Tensor, "pos"],
            Int[torch.Tensor, "1 pos"],
            Int[np.ndarray, "pos"],
            Int[np.ndarray, "1 pos"],
            list,
        ],
        prepend_bos: bool = True,
    ) -> List[str]:
        raise NotImplementedError(
            "to_str_tokens not supported for DecisionTransformer"
        )

    def to_single_token(self, string):
        raise NotImplementedError(
            "to_single_token not supported for DecisionTransformer"
        )

    def to_single_str_token(self, int_token: int) -> str:
        raise NotImplementedError(
            "to_single_str_token not supported for DecisionTransformer"
        )

    def get_token_position(
        self,
        single_token: Union[str, int],
        input: Union[
            str,
            Union[Float[torch.Tensor, "pos"], Float[torch.Tensor, "1 pos"]],
        ],
        mode="first",
        prepend_bos=True,
    ):
        raise NotImplementedError(
            "get_token_position not supported for DecisionTransformer"
        )

    def generate(
        self,
        input: Union[str, Float[torch.Tensor, "batch pos"]] = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        num_return_sequences: int = 1,
        use_past_kv_cache: bool = True,
        prepend_bos=True,
        return_type: Optional[str] = "input",
        verbose: bool = True,
    ) -> Float[torch.Tensor, "batch pos_plus_new_tokens"]:
        raise NotImplementedError(
            "generate not supported for DecisionTransformer"
        )

    def W_E(self) -> Float[torch.Tensor, "d_vocab d_model"]:
        raise NotImplementedError("W_E not supported for DecisionTransformer")

    def W_pos(self) -> Float[torch.Tensor, "d_vocab d_model"]:
        raise NotImplementedError(
            "W_pos not supported for DecisionTransformer"
        )

    def W_E_pos(self) -> Float[torch.Tensor, "d_vocab d_model"]:
        raise NotImplementedError(
            "W_E_pos not supported for DecisionTransformer"
        )

    def load_sample_training_dataset(self, **kwargs):
        raise NotImplementedError(
            "load_sample_training_dataset not supported for DecisionTransformer"
        )

    def sample_datapoint(
        self, tokenize=False
    ) -> Union[str, Float[torch.Tensor, "1 pos"]]:
        raise NotImplementedError(
            "sample_datapoint not supported for DecisionTransformer"
        )

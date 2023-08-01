"""Slight modification to vanilla GPT-style transformer to:
- Add a score embedding in the first input position
- Only predict player's actions"""

from typing import Dict, Union, Optional
from dataclasses import dataclass

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float32, Int64
from einops import rearrange, einsum, repeat

from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    utilities,
    utils,
)


class ScoreEmbed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_S: Float32[t.Tensor, "d_model"] = nn.Parameter(
            t.empty(self.cfg.d_model)
        )
        self.b_S: Float32[t.Tensor, "d_model"] = nn.Parameter(
            t.zeros(self.cfg.d_model)
        )

    def forward(
        self, score: Float32[t.Tensor, "batch"]
    ) -> Float32[t.Tensor, "batch d_model"]:
        """Embed a score vector into the transformer residual space"""
        return t.einsum("b,d->bd", score, self.W_S) + self.b_S


class ActionUnembed(nn.Module):
    """Unembeds the model's latent space into the next-action
    predictions at each relevant position. The resulting logits tensor
    has second player turns (or similar), not transformer positions."""

    def __init__(
        self,
        d_action: int,
        d_model: int,
        first_action_pos: int,
        action_pos_step: int,
    ):
        """Initialize the RSA unembedding module."""
        super().__init__()
        self.d_action = d_action
        self.first_action_pos = first_action_pos
        self.action_pos_step = action_pos_step
        self.W_U: Float32[tuple.Tensor, "d_model d_action"] = nn.Parameter(
            t.empty(d_model, d_action)
        )
        self.b_U: Float32[t.Tensor, "d_action"] = nn.Parameter(
            t.zeros(d_action)
        )

    def forward(
        self, residual: Float32[t.Tensor, "batch pos d_model"]
    ) -> Float32[t.Tensor, "batch turn d_action"]:
        """Unembed the model's latent space into the next action predictions"""
        return (
            einsum(
                residual[
                    :, (self.first_action_pos - 1) :: self.action_pos_step, :
                ],
                self.W_U,
                "batch turn d_model, d_model d_action -> batch turn d_action",
            )
            + self.b_U
        )


class IdentityVar(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        return x


@dataclass
class ScoreTransformerConfig:
    """Extra config values not present in the normal HookedTransformerConfig"""

    first_action_pos: int
    action_pos_step: int


class ScoreTransformer(HookedTransformer):
    def __init__(
        self,
        cfg: HookedTransformerConfig,
        st_cfg: ScoreTransformerConfig,
        move_to_device=True,
    ):
        super().__init__(cfg, move_to_device=move_to_device)

        # Override a few modules from the parent class
        self.first_device = utilities.devices.get_device_for_block_index(
            0, self.cfg
        )
        self.last_device = utilities.devices.get_device_for_block_index(
            self.cfg.n_layers - 1, self.cfg
        )
        # Create embedding module, move to correct device
        self.score_embed = ScoreEmbed(self.cfg).to(self.first_device)
        # Override a few modules from the parent class, moving to them
        # to the correct devices
        self.token_embed = self.embed
        self.pos_embed_copy = self.pos_embed
        self.embed = nn.Identity().to(self.first_device)
        self.pos_embed = IdentityVar().to(self.first_device)
        self.unembed = ActionUnembed(
            d_action=cfg.d_vocab_out,
            d_model=cfg.d_model,
            first_action_pos=st_cfg.first_action_pos,
            action_pos_step=st_cfg.action_pos_step,
        ).to(self.last_device)

        # Redo module initialization now that we've overridden some
        # (Will re-init all weights, keep that in mind for determinism)
        if self.cfg.init_weights:
            self.init_weights()

        # Redo setup call
        self.setup()

    def forward(
        self,
        tokens=Int64[t.Tensor, "batch pos"],
        scores=Float32[t.Tensor, "batch"],
        stop_at_layer: Optional[int] = None,
    ) -> Float32[
        t.Tensor, "batch turn d_action"
    ]:  # pylint: disable=arguments-differ
        """Forward pass through the model. Calls
        HookedTransformer.forward,
        but modifies some arguments, and does the token+score embedding
        first, passing in the embedded inputs tensors in place of the
        usual tokens tensor."""
        assert tokens.shape[1] <= self.cfg.n_ctx, "Input sequence too long!"
        # Embed the RSA inputs
        embeddings = self.token_embed(tokens) + self.pos_embed_copy(tokens)
        embeddings[:, 1, :] += self.score_embed(scores)
        # Run the HookedTransformer forward pass
        return super().forward(
            embeddings,
            return_type="logits",
            prepend_bos=False,
            stop_at_layer=stop_at_layer,
            past_kv_cache=None,
        )

    def loss_fn(
        self,
        logits: Float32[t.Tensor, "batch turn d_action"],
        actions: Int64[t.Tensor, "batch turn"],
        loss_mask: Optional[Float32[t.Tensor, "batch turn"]] = None,
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
            if loss_mask is not None:
                predicted_log_probs *= loss_mask
                loss_mean = -predicted_log_probs.sum() / loss_mask.sum()
            else:
                loss_mean = -predicted_log_probs.mean()
            return loss_mean

    def sample_next_action(
        self,
        tokens=Int64[t.Tensor, "batch pos"],
        scores=Float32[t.Tensor, "batch"],
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
    ) -> Int64[t.Tensor, "batch"]:
        """Do a forward pass and sample the next action. Provided tokens
        must end at a position for which a player action would be the
        expected next token."""
        # Get logits via a forward pass
        final_logits = self.forward(tokens, scores)[:, -1, :]
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

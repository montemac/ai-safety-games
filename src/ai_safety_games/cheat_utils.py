"""Functions for training and evaluating Cheat-playing models."""

from dataclasses import dataclass
import glob
import lzma
import os
import pickle
from typing import List, Callable, Dict, Any, Optional

import torch as t
from torch.utils.data import random_split
import numpy as np
from tqdm.auto import tqdm
from ai_safety_games import cheat, training
from ai_safety_games.ScoreTransformer import (
    ScoreTransformer,
    ScoreTransformerConfig,
    HookedTransformerConfig,
)


@dataclass(frozen=True, kw_only=True)
class TokenizedGames:
    """Tokenized games for training a Cheat-playing model."""

    tokens: t.Tensor
    scores: t.Tensor
    seq_lens: t.Tensor
    loss_mask: t.Tensor


@dataclass(frozen=True, kw_only=True)
class CheatTrainingConfig:
    """Config for training a Cheat-playing model."""

    dataset_folder: str
    game_filter: Optional[Callable[Dict[str, List[Any]], List[int]]] = None
    embedding_type: str = "tokens"  # TODO: implement this?
    cached_game_data: Optional[TokenizedGames] = None
    train_fraction: float
    n_layers: int
    d_model: int
    d_head: int
    attn_only: bool
    device: str
    training_mins: int
    batch_size: int
    lr: float
    lr_schedule: str
    weight_decay: float
    log_period: int
    seed: int


def train(config: CheatTrainingConfig):
    """Function to train a Cheat-playing model."""
    # Load dataset config info
    with open(os.path.join(config.dataset_folder, "config.pkl"), "rb") as file:
        config_dict = pickle.load(file)
        game_config = cheat.CheatConfig(**config_dict["game.config"])
        game = cheat.CheatGame(game_config)

    # Load summary info
    with open(
        os.path.join(config.dataset_folder, "summary.pkl"), "rb"
    ) as file:
        summary_lists = pickle.load(file)

    # Filter the games, if provided
    if config.game_filter is not None:
        game_inds = config.game_filter(summary_lists)
    else:
        game_inds = range(len(summary_lists["turn_cnt"]))
    game_fns = [
        os.path.join(config.dataset_folder, f"game_{game_idx}.pkl")
        for game_idx in game_inds
    ]

    # Get the token vocab
    vocab, player_action_vocab = game.get_token_vocab()
    vocab_strs = {idx: tok for tok, idx in vocab.items()}

    # Check if we have been passed pre-cached game data
    if config.cached_game_data is not None:
        # Load games and convert to token tensors plus scores
        tokens_list = []
        scores_list = []
        for game_idx, filename in enumerate(tqdm(game_fns)):
            with lzma.open(filename, "rb") as file:
                game_results = pickle.load(file)
                # Rebuild state history
                state_history = [
                    cheat.CheatState(**state_dict)
                    for state_dict in game_results["state_history_list"]
                ]
                # Calculate scores from state history (there was a bug in scores
                # calculation before so don't use loaded scores)
                hand_sizes = np.array(
                    [len(hand) for hand in state_history[-1].hands]
                )
                scores = -hand_sizes
                next_best_score = np.max(scores[scores != 0])
                scores[scores == 0] = -next_best_score
                # Get token sequences
                tokens_this = cheat.get_seqs_from_state_history(
                    game=game, vocab=vocab, state_history=state_history
                )
                tokens_list.extend([row for row in tokens_this])
                scores_list.append(scores)

        # Convert to single tokens tensor, scores tensor, and seq lengths tensor
        # Use pad_sequence to pad to the max length of any game
        tokens_all = t.nn.utils.rnn.pad_sequence(
            tokens_list, batch_first=True, padding_value=0
        ).to(config.device)
        scores_all = t.tensor(
            np.array(scores_list).flatten(), dtype=t.float32
        ).to(config.device)
        seq_lens_all = t.tensor(
            [len(toks) for toks in tokens_list], dtype=t.int64
        ).to(config.device)

        # Positions corresponding to actions
        first_action_pos = (
            (game_config.num_players - 1) * 2 + 2 + game.config.num_ranks
        )
        action_pos_step = game_config.num_players * 2 + game.config.num_ranks

        # Turn sequence lengths into a loss mask
        loss_mask = t.zeros_like(tokens_all, dtype=t.float32)
        for idx, seq_len in enumerate(seq_lens_all):
            loss_mask[idx, :seq_len] = 1
        loss_mask = loss_mask[:, first_action_pos::action_pos_step]

        # Store all in a dataset
        game_data = TokenizedGames(
            tokens=tokens_all,
            scores=scores_all,
            seq_lens=seq_lens_all,
            loss_mask=loss_mask,
        )
    else:
        game_data = config.cached_game_data

    # Training loop using library function

    # Split data into train and test sets
    # TODO: probably a simpler way to do this
    generator = t.Generator().manual_seed(config.seed)
    train_inds, test_inds = [
        t.tensor(subset)
        for subset in random_split(
            range(game_data.tokens.shape[0]),
            [config.train_fraction, (1 - config.train_fraction)],
            generator=generator,
        )
    ]

    # Initialize a simple test model
    model = ScoreTransformer(
        cfg=HookedTransformerConfig(
            n_layers=config.n_layers,
            d_model=config.d_model,
            d_head=config.d_head,
            d_vocab=len(vocab),
            d_vocab_out=len(player_action_vocab),
            act_fn="relu",
            device=config.device,
            seed=config.seed,
            n_ctx=seq_lens_all.max().item(),
            attn_only=config.attn_only,
        ),
        st_cfg=ScoreTransformerConfig(
            first_action_pos=first_action_pos, action_pos_step=action_pos_step
        ),
    )

    # Standard test function
    test_func = training.make_standard_test_func(
        test_data=training.TrainingTensors(
            inputs=[game_data.tokens[test_inds], game_data.scores[test_inds]],
            output=game_data.tokens[
                test_inds, first_action_pos::action_pos_step
            ]
            .detach()
            .clone(),
            loss_mask=game_data.loss_mask[test_inds],
        ),
        test_batch_size=config.batch_size,
    )

    # Train!
    results = training.train_custom_transformer(
        model=model,
        config=training.TrainingConfig(
            project_name="cheat",
            training_mins=config.training_mins,
            batch_size=config.batch_size,
            lr=config.lr,
            weight_decay=config.weight_decay,
            log_period=config.log_period,
            seed=config.seed,
            save_results=True,
        ),
        training_data=training.TrainingTensors(
            inputs=[
                game_data.tokens[train_inds],
                game_data.scores[train_inds],
            ],
            output=game_data.tokens[
                train_inds, first_action_pos::action_pos_step
            ]
            .detach()
            .clone(),
            loss_mask=game_data.loss_mask[train_inds],
        ),
        test_func=test_func,
    )

    return results, game_data

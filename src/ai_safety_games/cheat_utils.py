"""Functions for training and evaluating Cheat-playing models."""

from dataclasses import dataclass, asdict
import glob
import lzma
import os
import pickle
import datetime
from typing import List, Callable, Dict, Any, Optional, Tuple

import torch as t
from torch import nn
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
    device: str
    game_filter: Optional[Callable[Dict[str, List[Any]], List[int]]] = None
    embedding_type: str = "tokens"  # TODO: implement this?
    cached_game_data: Optional[TokenizedGames] = None
    train_fraction: float
    n_layers: int
    d_model: int
    d_head: int
    attn_only: bool
    epochs: int
    batch_size: int
    lr: float
    lr_schedule: Optional[Tuple[str, Dict[str, Any]]] = None
    weight_decay: float
    log_period: int
    seed: int
    test_goal_scores: Optional[List[int]] = None
    test_player_inds: Optional[List[int]] = None
    num_test_games: int = 100


def load_game_data(
    dataset_folder: str,
    device: str,
    game_filter: Optional[Callable[Dict[str, List[Any]], List[int]]] = None,
) -> TokenizedGames:
    """Load game data from disk, and optinoally filter."""
    # Load dataset config info
    with open(os.path.join(dataset_folder, "config.pkl"), "rb") as file:
        config_dict = pickle.load(file)
    game_config = cheat.CheatConfig(**config_dict["game.config"])
    game = cheat.CheatGame(game_config)

    # Load summary info
    with open(os.path.join(dataset_folder, "summary.pkl"), "rb") as file:
        summary_lists = pickle.load(file)

    # Filter the games, if provided
    if game_filter is not None:
        game_inds = game_filter(summary_lists)
    else:
        game_inds = range(len(summary_lists["turn_cnt"]))
    game_fns = [
        os.path.join(dataset_folder, f"game_{game_idx}.pkl")
        for game_idx in game_inds
    ]

    # Get the token vocab
    vocab, player_action_vocab = game.get_token_vocab()
    vocab_strs = {idx: tok for tok, idx in vocab.items()}

    # Positions corresponding to actions
    first_action_pos = (
        (game_config.num_players - 1) * 2 + 2 + game.config.num_ranks
    )
    action_pos_step = game_config.num_players * 2 + game.config.num_ranks

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
    ).to(device)
    scores_all = t.tensor(np.array(scores_list).flatten(), dtype=t.float32).to(
        device
    )
    seq_lens_all = t.tensor(
        [len(toks) for toks in tokens_list], dtype=t.int64
    ).to(device)

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
    return game_data


def train(config: CheatTrainingConfig):
    """Function to train a Cheat-playing model."""
    # Load dataset config info
    with open(os.path.join(config.dataset_folder, "config.pkl"), "rb") as file:
        config_dict = pickle.load(file)
    game_config = cheat.CheatConfig(**config_dict["game.config"])
    game = cheat.CheatGame(game_config)
    # Load player info
    players_all = []
    for class_str, player_vars in config_dict["players"]:
        class_name = class_str.split("'")[1].split(".")[-1]
        if class_name == "NaiveCheatPlayer":
            player = cheat.NaiveCheatPlayer()
        elif class_name == "XRayCheatPlayer":
            player = cheat.XRayCheatPlayer(
                probs_table=player_vars["probs_table"]
            )
        elif class_name == "AdaptiveCheatPlayer":
            player = cheat.AdaptiveCheatPlayer(
                max_call_prob=player_vars["max_call_prob"],
                max_cheat_prob=player_vars["max_cheat_prob"],
                is_xray=player_vars["is_xray"],
            )
        players_all.append(player)

    # Filter out any players if provided
    if config.test_player_inds is not None:
        test_players = [players_all[idx] for idx in config.test_player_inds]
    else:
        test_players = players_all

    # Load summary info
    with open(
        os.path.join(config.dataset_folder, "summary.pkl"), "rb"
    ) as file:
        summary_lists = pickle.load(file)

    # Get the token vocab
    vocab, player_action_vocab = game.get_token_vocab()

    # Positions corresponding to actions
    first_action_pos = (
        (game_config.num_players - 1) * 2 + 2 + game.config.num_ranks
    )
    action_pos_step = game_config.num_players * 2 + game.config.num_ranks

    # Check if we have been passed pre-cached game data
    if config.cached_game_data is None:
        game_data = load_game_data(
            dataset_folder=config.dataset_folder,
            device=config.device,
            game_filter=config.game_filter,
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
            n_ctx=game_data.seq_lens.max().item(),
            attn_only=config.attn_only,
        ),
        st_cfg=ScoreTransformerConfig(
            first_action_pos=first_action_pos, action_pos_step=action_pos_step
        ),
    )

    # Standard test function
    test_loss_func = training.make_standard_test_func(
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

    parent_config = config

    def test_func(
        model: nn.Module, config: training.TrainingConfig, test_idx: int
    ):
        """Test function that runs some test games and gets loss on test
        dataset"""
        # Get test loss
        test_results = test_loss_func(model, config, test_idx)
        # Play some test games
        if parent_config.test_goal_scores is None:
            test_goal_scores = [2]
        else:
            test_goal_scores = parent_config.test_goal_scores
        for goal_score in test_goal_scores:
            test_margins = run_test_games(
                model=model,
                game_config=game_config,
                num_games=parent_config.num_test_games,
                goal_score=goal_score,
                max_turns=max(summary_lists["turn_cnt"]),
                players=test_players,
                seed=parent_config.seed,
            )
            test_results[f"test_margin_mean_goal_{goal_score}"] = np.mean(
                test_margins
            )
            test_results[f"test_margin_std_goal_{goal_score}"] = np.std(
                test_margins
            )
        return test_results

    # Train!
    results = training.train_custom_transformer(
        model=model,
        config=training.TrainingConfig(
            project_name="cheat",
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            lr_schedule=config.lr_schedule,
            weight_decay=config.weight_decay,
            log_period=config.log_period,
            seed=config.seed,
            save_results=False,
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

    # Store results
    # Create a timestamped output directory
    output_dir = os.path.join(
        "cheat_train_results",
        datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "results.pkl"), "wb") as file:
        pickle.dump(
            {"config": asdict(config), "training_results": asdict(results)},
            file,
        )

    # Return
    return results, game_data


def scores_to_margins(scores):
    """Convert scores to victory margins."""
    try:
        best_nonwinning_score = max([score for score in scores if score < 0])
    except ValueError:
        return scores
    return [
        score if score < 0 else score - best_nonwinning_score
        for score in scores
    ]


def run_test_games(
    model: ScoreTransformer,
    game_config: cheat.CheatConfig,
    num_games: int,
    goal_score: int,
    max_turns: int,
    players: List[cheat.CheatPlayer],
    seed: int,
):
    """Run some test games to see how the model performs."""
    rng = np.random.default_rng(seed=seed)

    # Create the game
    game = cheat.CheatGame(config=game_config)
    vocab, _ = game.get_token_vocab()

    model_margins = []
    for game_idx in range(num_games):
        # Create a list of players with the model-based player in the first
        # position, then other randomly-selected players
        players_this = [
            cheat.ScoreTransformerCheatPlayer(
                model=model, vocab=vocab, goal_score=goal_score
            ),
            *rng.choice(
                players, size=game_config.num_players - 1, replace=True
            ),
        ]

        # Run the game
        scores, winning_player, turn_cnt = cheat.run(
            game=game,
            players=players_this,
            max_turns=max_turns,
            seed=rng.integers(1e6),
        )

        # Calculate victory margins
        margins_this = scores_to_margins(scores)

        model_margins.append(margins_this[0])

    return np.array(model_margins)

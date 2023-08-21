"""Interpret a cheat model."""
# %%
# Imports, etc.

import pickle
import datetime
import glob
import os

# TEMP
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import pandas as pd
import torch as t
from tqdm.auto import tqdm
import plotly.express as px
from einops import rearrange, einsum

from ai_safety_games import cheat, utils, training, cheat_utils
from ai_safety_games.ScoreTransformer import ScoreTransformer

utils.enable_ipython_reload()

# Disable gradients
_ = t.set_grad_enabled(False)


# %%
# Load some stuff about the dataset, and the model to test

# 4L attn-only model
# TRAINING_RESULTS_FN = "cheat_results/20230801T095350/results.pkl"
# 1L attn-only model
# TRAINING_RESULTS_FN = "cheat_results/20230801T105951/results.pkl"
# 4L full model
# TRAINING_RESULTS_FN = "cheat_results/20230801T111728/results.pkl"
# 4L full model, separate score and BOG embeddings
# TRAINING_RESULTS_FN = "cheat_results/20230801T123246/results.pkl"
# 8L full model, separate score and BOG embeddings
# TRAINING_RESULTS_FN = "cheat_results/20230801T130838/results.pkl"
#
# TRAINING_RESULTS_FN = "cheat_train_results/20230815T153630/results.pkl"
# Trained on win only
# TRAINING_RESULTS_FN = "cheat_train_results/20230815T234841/results.pkl"
# Small model, trained only on naive and 0.25/1.0 adaptive
TRAINING_RESULTS_FN = "cheat_train_results/20230817T151856/results.pkl"

# Load model

# TODO: fix this problem with loading models!
# AttributeError: Can't get attribute 'game_filter' on
game_filter = None

with open(TRAINING_RESULTS_FN, "rb") as file:
    results_all = pickle.load(file)
config = cheat_utils.CheatTrainingConfig(**results_all["config"])
results = training.TrainingResults(**results_all["training_results"])
model = results.model

# Do some interp-friendly processing of weights
model.process_weights_(
    fold_ln=True, center_writing_weights=False, center_unembed=True
)

with open(os.path.join(config.dataset_folder, "config.pkl"), "rb") as file:
    config_dict = pickle.load(file)
    game_config = cheat.CheatConfig(**config_dict["game.config"])
    # Load players (don't save classes in case they have changed a bit
    # and loading breaks)
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
            )
        players_all.append(player)

game = cheat.CheatGame(config=game_config)
vocab, action_vocab = game.get_token_vocab()
vocab_str = {idx: token_str for token_str, idx in vocab.items()}


# %%
# Inspect some of the model weights
W_QK = einsum(
    model.blocks[0].attn.W_Q,
    model.blocks[0].attn.W_K,
    "h dmq dh, h dmk dh -> h dmq dmk",
)
QK = einsum(
    model.token_embed.W_E,
    W_QK,
    model.token_embed.W_E,
    "tq dmq, h dmq dmk, tk dmk -> h tq tk",
)
W_OV = einsum(
    model.blocks[0].attn.W_O,
    model.blocks[0].attn.W_V,
    "h dh dmo, h dmv dh -> h dmo dmv",
)
OV = einsum(
    model.unembed.W_U,
    W_OV,
    model.token_embed.W_E,
    "dmo da, h dmo dmv, tv dmv -> h da tv",
)
# Only look at possible destination tokens that could preceed our
# action, i.e. number of final rank cards
dst_inds = t.tensor(
    [
        vocab[f"hand_{num}x{game_config.num_ranks-1}"]
        for num in range(game_config.num_suits + 1)
    ]
)
act_inds = t.tensor(list(action_vocab.keys()))

QKf = QK[:, dst_inds, :].cpu().numpy()
OVf = OV[:, act_inds, :].cpu().numpy()


# %%
# Run a game, then turn the history into a token sequence and run it through the model
SEED = 0
GOAL_SCORE = 5

rng = np.random.default_rng(seed=SEED)

players_can_use = [players_all[idx] for idx in [0, 3]]

# Create a list of players with the model-based player in the first
# position, then two other randomly-selected players
player_inds_this = rng.choice(len(players_can_use), size=2, replace=True)
players = [
    cheat.ScoreTransformerCheatPlayer(
        model=model,
        vocab=vocab,
        goal_score=GOAL_SCORE,
        temperature=0,
    ),
    *[players_can_use[idx] for idx in player_inds_this],
]

# Run the game
seed = rng.integers(1e6)
scores, winning_player, turn_cnt = cheat.run(
    game=game,
    players=players,
    max_turns=40,
    seed=seed,
)
print(game.print_state_history())

# Turn history into a token sequence
tokens = cheat.get_seqs_from_state_history(
    game=game,
    vocab=vocab,
    state_history=game.state_history,
    players_to_return=[0],
)

# Get logits
action_logits = model(
    tokens=tokens,
    scores=t.tensor([GOAL_SCORE]).to(model.cfg.device),
).squeeze()
action_probs = t.softmax(action_logits, dim=1)
sort_inds = action_logits.argsort(dim=1)
# Get the tokens corresponding to the top K actions
top_k = 10
top_k_tokens = pd.DataFrame(
    [
        [
            action_probs[i, idx].item()
            # vocab_str[idx]
            for idx in sort_inds[i, -top_k:].cpu().numpy()[::-1]
        ]
        for i in range(sort_inds.shape[0])
    ]
).T
top_k_tokens

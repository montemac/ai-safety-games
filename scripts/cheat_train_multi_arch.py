# %%
# Imports, etc
import os
from typing import Dict, List, Any
import pickle

# TEMP
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import pandas as pd
import torch as t
import plotly.express as px
from tqdm.auto import tqdm

from ai_safety_games import cheat_utils, utils

utils.enable_ipython_reload()

_ = t.set_grad_enabled(True)


# %%
# Load game data

# Setup constants
# DATASET_FOLDER = "../datasets/random_dataset_20230731T001342"
DATASET_FOLDER = "../datasets/random_dataset_20230815T235622"
DEVICE = "cuda:0"
SEQUENCE_MODE = "tokens_score"

INCLUDED_PLAYERS = [0, 3]

INCLUDE_HAND_END = True


def game_filter(summary_lists: Dict[str, List[Any]]) -> List[int]:
    """Filter out games that don't match criteria"""
    inds = []
    for idx, player_inds in enumerate(summary_lists["player_indices"]):
        if all([player_ind in INCLUDED_PLAYERS for player_ind in player_inds]):
            inds.append(idx)
    return inds


game_data = cheat_utils.load_game_data(
    dataset_folder=DATASET_FOLDER,
    sequence_mode=SEQUENCE_MODE,
    game_filter=game_filter,
    device=DEVICE,
    include_hand_end=INCLUDE_HAND_END,
)

# %%
# Train models!
ARCH_PARAMS = [(32, 32)]

for d_model, d_head in tqdm(ARCH_PARAMS):
    # Train using new high-level function
    results, game_data, test_func = cheat_utils.train(
        cheat_utils.CheatTrainingConfig(
            dataset_folder=DATASET_FOLDER,
            sequence_mode=SEQUENCE_MODE,
            game_filter=game_filter,
            device=DEVICE,
            cached_game_data=game_data,
            train_fraction=0.99,
            n_layers=1,
            d_model=d_model,
            d_head=d_head,
            attn_only=True,
            max_turns=40,
            include_hand_end=INCLUDE_HAND_END,
            epochs=500,
            # epochs=int(10 * 125000 / len(game_data.loaded_game_inds)),
            batch_size=1000,
            lr=0.001,
            # lr_schedule=("cosine_with_warmup", {"warmup_fraction": 0.05}),
            lr_schedule=None,
            weight_decay=0,
            log_period=500000,
            seed=1,
            test_player_inds=INCLUDED_PLAYERS,
            test_goal_scores=[0, 5],
            cheat_penalty_weight=0,
            cheat_penalty_apply_prob=0,
            cheat_penalty_min_prob=0.1,
        )
    )

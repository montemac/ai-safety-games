"""Test a previously trained cheat model on new games."""
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

from ai_safety_games import cheat, utils, models

utils.enable_ipython_reload()

# %%
# Create some random players and a model-based player, run the game
# Load the model-based player
MODEL_FN = "cheat_train_output/20230710T080356/model.pkl"
with open(MODEL_FN, "rb") as file:
    model = pickle.load(file)

PROBS_TABLE = pd.DataFrame(
    {
        "can_play": {
            "pass": 0.0,
            "call": 0.0,
            "cheat": 0.0,
            "play": 1.0,
        },
        "cannot_play": {
            "pass": 0.8,
            "call": 0.0,
            "cheat": 0.2,
            "play": 0.0,
        },
    }
)

# List of players
players = [
    cheat.ModelCheatPlayer(model=model, goal_score=0),
    cheat.RandomCheatPlayer(probs_table=PROBS_TABLE),
    cheat.RandomCheatPlayer(probs_table=PROBS_TABLE),
]

# Create the game
game = cheat.CheatGame(
    config=cheat.CheatConfig(num_ranks=13, num_suits=4, seed=SEED)
)

# Run the games
scores_list = []
for seed in tqdm(range(100)):
    scores, winning_player, turn_cnt = cheat.run(
        game=game,
        players=players,
        max_turns=model.dt_cfg.n_timesteps * game.config.num_players,
        seed=seed,
    )
    scores_list.append(scores)
scores_df = pd.DataFrame(scores_list)
print(scores_df.mean())

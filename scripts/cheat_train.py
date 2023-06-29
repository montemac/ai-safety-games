# %%
# Imports, etc.

import pickle
import datetime
import glob
import os

from sortedcontainers import SortedList
from collections import defaultdict
import numpy as np
import pandas as pd
import torch as t
from tqdm.auto import tqdm
import plotly.express as px

from ai_safety_games import cheat, utils

utils.enable_ipython_reload()

# %%
# Load a dataset into memory
FOLDER = "../datasets/random_dataset_20230628T150122"
DEVICE = "cuda:1"

# Load config info
with open(os.path.join(FOLDER, "config.pkl"), "rb") as file:
    config_dict = pickle.load(file)
    game_config = cheat.CheatConfig(**config_dict["game.config"])
    players_all = [
        cheat.RandomCheatPlayer(player_config["probs_table"])
        for player_config in config_dict["players"]
    ]

# Load summary data
with open(os.path.join(FOLDER, "summary.pkl"), "rb") as file:
    summary_lists = pickle.load(file)

# Load all the RSA tensors into the GPU, in a list by game index
# (We don't mash them all together because we want to train on one game at a time)
rewards_tensors = []
state_tensors = []
action_tensors = []
for filename in tqdm(sorted(glob.glob(os.path.join(FOLDER, "rsas_*.pkl")))):
    with open(filename, "rb") as file:
        reward, state, action = pickle.load(file)
    rewards_tensors.append(reward.to(DEVICE))
    state_tensors.append(state.to(DEVICE))
    action_tensors.append(action.to(DEVICE))


# %%
# Run the training loop

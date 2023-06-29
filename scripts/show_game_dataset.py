# %%
# Imports, etc.

import pickle
import datetime
import os

from sortedcontainers import SortedList
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import plotly.express as px

from ai_safety_games import cheat, utils

utils.enable_ipython_reload()


# %%
# Load the results, then process into a dataframe for analysis
FOLDER = os.path.join("../datasets/random_dataset_20230628T112736")

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

# Create player probability arrays
prob_arrays = []
for player in players_all:
    probs = player.probs_table.T.stack()
    probs.index = [f"prob-{level1}-{level2}" for level1, level2 in probs.index]
    prob_arrays.append(probs)

# Assemble into a dataframe
# Probs first
player_data = pd.DataFrame(prob_arrays)

# Games by player
player_data["game_count"] = (
    pd.Series(np.array(summary_lists["player_indices"]).flatten())
    .value_counts()
    .sort_index()
)

# Wins by player, and win rate
player_data["win_count"] = (
    pd.Series(summary_lists["winning_player"]).value_counts().sort_index()
)
player_data["win_rate"] = player_data["win_count"] / player_data["game_count"]

# Group by each probability, and average the win rate, then concatentate
# into a single dataframe
win_rates_df = (
    player_data.reset_index()
    .melt(
        id_vars=["win_rate", "index"],
        value_vars=prob_cols,
        var_name="prob_name",
        value_name="prob",
    )
    .join(player_data[prob_cols], "index")
)


# %%
# Plot stuff

px.scatter(
    win_rates_df,
    x="prob",
    y="win_rate",
    facet_col="prob_name",
    facet_col_wrap=4,
    opacity=0.1,
    hover_data=prob_cols,
).show()

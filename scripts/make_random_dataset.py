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
# Create a set of random players

# Define the probability posibilities
PROB_POINTS = {
    ("can_play", "pass"): [0, 0.1],
    ("can_play", "call"): [0, 0.05, 0.1, 0.2],
    ("can_play", "cheat"): [0, 0.1, 0.2, 0.4],
    ("can_play", "play"): [1],
    ("cannot_play", "pass"): [1],
    ("cannot_play", "call"): [0, 0.05, 0.1, 0.2],
    ("cannot_play", "cheat"): [0, 0.2, 0.4, 0.8],
    ("cannot_play", "play"): [0],
}

# Create an array with each row representing a choice for each probability
prob_arrays = np.concatenate(
    [
        arr.flatten()[:, None]
        for arr in np.meshgrid(*PROB_POINTS.values(), indexing="ij")
    ],
    axis=1,
)

# Turn each row into a player
players_all = []
for probs in prob_arrays:
    probs_dict = {"can_play": {}, "cannot_play": {}}
    for (can_play, action), prob in zip(PROB_POINTS.keys(), probs):
        probs_dict[can_play][action] = prob
    probs_table = pd.DataFrame(probs_dict)
    players_all.append(cheat.RandomCheatPlayer(probs_table=probs_table))


# %%
# Run a large number of games with incrementing seeds, randomly
# selecting players for each game, and recording the results (winner and
# scores)
NUM_GAMES = 10000

# Create the game
game = cheat.CheatGame(
    config=cheat.CheatConfig(num_ranks=13, num_suits=4, seed=0)
)

# Create a folder for this run
folder = f"../datasets/random_dataset_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}"
os.mkdir(folder)

# Run the games, storing results of each game as we go
for seed in tqdm(range(NUM_GAMES)):
    # Pick the players
    rng = np.random.default_rng(seed=seed)
    player_indices = rng.choice(len(players_all), size=game.config.num_players)
    players = [players_all[idx] for idx in player_indices]

    # Run the game
    scores, winning_player, turn_cnt = cheat.run(
        game=game,
        players=players,
        max_turns=1000,
        seed=seed,
    )
    # Store results
    results = {
        "seed": seed,
        "turn_cnt": turn_cnt,
        "player_indices": player_indices,
        "winning_player": player_indices[winning_player],
        "scores": {idx: score for idx, score in zip(player_indices, scores)},
        "state_history_list": game.state_history,
    }
    with open(os.path.join(folder, f"{seed}.pkl"), "wb") as file:
        pickle.dump(results, file)


# %%
# Load the results, then process into a dataframe for analysis
# FN = "random_dataset-20230626T141943.pkl"
# with open(FN, "rb") as file:
#     results = pickle.load(file)

# # Probs first
# player_data = pd.DataFrame(
#     prob_arrays,
#     columns=[f"prob-{key[0]}-{key[1]}" for key in PROB_POINTS.keys()],
# )

# # Games by player
# player_data["game_count"] = (
#     pd.Series(np.array(results["players_list"]).flatten())
#     .value_counts()
#     .sort_index()
# )

# # Wins by player, and win rate
# player_data["win_count"] = (
#     pd.Series(results["winning_player_list"]).value_counts().sort_index()
# )
# player_data["win_rate"] = player_data["win_count"] / player_data["game_count"]

# # Group by each probability, and average the win rate, then concatentate
# # into a single dataframe
# win_rates_list = []
# for key in PROB_POINTS.keys():
#     col_name = f"prob-{key[0]}-{key[1]}"
#     win_rate_this = (
#         player_data.groupby(col_name).mean()["win_rate"].reset_index()
#     ).set_axis(["prob_value", "win_rate"], axis=1)
#     win_rate_this["prob_key"] = col_name
#     win_rates_list.append(win_rate_this)
# win_rates_df = pd.concat(win_rates_list, axis=0, ignore_index=True)

# px.line(
#     win_rates_df,
#     x="prob_value",
#     y="win_rate",
#     facet_col="prob_key",
#     facet_col_wrap=4,
# ).show()

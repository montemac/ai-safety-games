# %%
# Imports, etc.

import pickle
import datetime
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

# Store the game config and player probabilities
with open(os.path.join(folder, "config.pkl"), "wb") as file:
    pickle.dump(
        {
            "game.config": vars(game.config),
            "players": [vars(player) for player in players_all],
        },
        file,
    )

# Run the games, storing results of each game as we go
summary_lists = defaultdict(list)
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
    with open(os.path.join(folder, f"game_{seed}.pkl"), "wb") as file:
        pickle.dump(results, file)

    # Make reward-state-action sequences for this game
    rsas_by_player = defaultdict(list)
    for turn in range(len(game.state_history)):
        # Get the RSA tuple for this turn
        player_num, rsa = cheat.get_rsa(
            states=game.state_history,
            turn=turn,
            scores=scores,
            game=game,
        )
        # Store this RSA tuple
        rsas_by_player[player_num].append(rsa)

    # Store the RSA sequences in a single set of tensors
    rsa_tensors = []
    for idx in range(len(rsas_by_player[0][0])):
        rsa_tensors.append(
            t.tensor(
                np.array(
                    [
                        np.array(
                            [rsa[idx] for rsa in rsas_by_player[player_num]]
                        )
                        for player_num in rsas_by_player.keys()
                    ]
                ),
                dtype=t.float32,
            )
        )
    with open(os.path.join(folder, f"rsas_{seed}.pkl"), "wb") as file:
        pickle.dump(rsa_tensors, file)

    # Store results useful for summary
    for key in results:
        if key not in ["state_history_list"]:
            summary_lists[key].append(results[key])

# Save summary results list
with open(os.path.join(folder, "summary.pkl"), "wb") as file:
    pickle.dump(dict(summary_lists), file)

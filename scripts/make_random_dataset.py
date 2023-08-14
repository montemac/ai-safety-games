# %%
# Imports, etc.

import pickle
import datetime
import os
import lzma

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
PLAYER_SETUP = [
    # Never cheats or calls
    (cheat.NaiveCheatPlayer(), 1.0),
    # Naive player with X-ray vision, supervisor role
    (
        cheat.XRayCheatPlayer(
            pd.DataFrame(
                {
                    "can_play": {"pass": 0, "call": 0, "cheat": 0, "play": 1},
                    "cannot_play": {
                        "pass": 1,
                        "call": 0,
                        "cheat": 0,
                        "play": 0,
                    },
                }
            )
        ),
        1.0,
    ),
    # Mild adaptive cheater, never calls
    (
        cheat.AdaptiveCheatPlayer(
            max_call_prob=0.0,
            max_cheat_prob=0.25,
        ),
        1.0,
    ),
    # Heavy adaptive cheater, never calls
    (
        cheat.AdaptiveCheatPlayer(
            max_call_prob=0.0,
            max_cheat_prob=1.0,
        ),
        1.0,
    ),
    # Mild adaptive cheater, sometimes calls
    (
        cheat.AdaptiveCheatPlayer(
            max_call_prob=0.25,
            max_cheat_prob=0.25,
        ),
        1.0,
    ),
    # Heavy adaptive cheater, sometimes calls
    (
        cheat.AdaptiveCheatPlayer(
            max_call_prob=0.25,
            max_cheat_prob=1.0,
        ),
        1.0,
    ),
    # Mild adaptive cheater, x-ray
    (
        cheat.AdaptiveCheatPlayer(
            max_call_prob=0.0,
            max_cheat_prob=0.25,
            is_xray=True,
        ),
        1.0,
    ),
    # Heavy adaptive cheater, x-ray
    (
        cheat.AdaptiveCheatPlayer(
            max_call_prob=0.0,
            max_cheat_prob=1.0,
            is_xray=True,
        ),
        1.0,
    ),
]

PLAYERS = [player for player, _ in PLAYER_SETUP]
PLAYER_WEIGHTS = np.array([weight for _, weight in PLAYER_SETUP])
PLAYER_WEIGHTS /= PLAYER_WEIGHTS.sum()


# %%
# Run a large number of games with incrementing seeds, randomly
# selecting players for each game, and recording the results (winner and
# scores)
NUM_GAMES = 5000000
MAX_TURNS = 40

# Create the game
game = cheat.CheatGame(
    config=cheat.CheatConfig(num_ranks=8, num_suits=2, seed=0)
)

# Create a folder for this run
folder = f"../datasets/random_dataset_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}"
os.mkdir(folder)

# Store the game config and player probabilities
with open(os.path.join(folder, "config.pkl"), "wb") as file:
    pickle.dump(
        {
            "game.config": vars(game.config),
            "players": [
                (str(type(player)), vars(player)) for player in PLAYERS
            ],
        },
        file,
    )

# Run the games, storing results of each game as we go
summary_lists = defaultdict(list)
for game_idx in tqdm(range(NUM_GAMES)):
    # Pick the players
    rng = np.random.default_rng(seed=game_idx)
    player_indices = rng.choice(
        len(PLAYERS), size=game.config.num_players, p=PLAYER_WEIGHTS
    )
    players = [PLAYERS[idx] for idx in player_indices]

    # Run the game
    scores, winning_player, turn_cnt = cheat.run(
        game=game,
        players=players,
        max_turns=MAX_TURNS,
        seed=rng.integers(1e6),
        # verbose=True,
    )

    # Store results
    results = {
        "game_idx": game_idx,
        "turn_cnt": turn_cnt,
        "player_indices": player_indices,
        "winning_player": player_indices[winning_player]
        if winning_player is not None
        else None,
        "scores": list(zip(player_indices, scores)),
        "state_history_list": [vars(state) for state in game.state_history],
    }
    with lzma.open(os.path.join(folder, f"game_{game_idx}.pkl"), "wb") as file:
        pickle.dump(results, file)

    # Store results useful for summary
    for key in results:
        if key not in ["state_history_list"]:
            summary_lists[key].append(results[key])

# Save summary results list
with open(os.path.join(folder, "summary.pkl"), "wb") as file:
    pickle.dump(dict(summary_lists), file)

# %%

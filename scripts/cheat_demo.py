# %%
# Imports, etc.

from sortedcontainers import SortedList
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import plotly.express as px

from ai_safety_games import cheat, utils

utils.enable_ipython_reload()

# %%
# Run some random games
SEED = 0
rng = np.random.default_rng(seed=SEED)

# Create the game
game = cheat.CheatGame(
    config=cheat.CheatConfig(num_ranks=13, num_suits=4, seed=SEED)
)

honest_player = cheat.RandomCheatPlayer(
    probs_table=pd.DataFrame(
        {
            "can_play": {
                "pass": 0,
                "call": 0,
                "cheat": 0,
                "play": 1,
            },
            "cannot_play": {
                "pass": 1,
                "call": 0,
                "cheat": 0,
                "play": 0,
            },
        }
    )
)

# Make an "honest" player and some "cheating" players
players = [honest_player] + [
    cheat.RandomCheatPlayer(
        probs_table=pd.DataFrame(
            {
                "can_play": {
                    "pass": 0.0,
                    "call": 0.1,
                    "cheat": 0.5,
                    "play": 1,
                },
                "cannot_play": {
                    "pass": 1,
                    "call": 0.1,
                    "cheat": 0.5,
                    "play": 0,
                },
            }
        )
    )
    for _ in range(game.config.num_players - 1)
]

# Run a few games
NUM_GAMES = 1000

winning_players = []
turn_cnts = []
for seed in tqdm(range(NUM_GAMES)):
    # Run the game
    scores, winning_player, turn_cnt = cheat.run(
        game=game,
        players=players,
        max_turns=1000,
        seed=seed,
    )
    winning_players.append(winning_player)
    turn_cnts.append(turn_cnt)

# Plot results
px.histogram(winning_players, nbins=game.config.num_players).show()
px.histogram(turn_cnts).show()

# %%

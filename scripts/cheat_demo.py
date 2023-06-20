# %%
# Imports, etc.

from sortedcontainers import SortedList
import pandas as pd
from ai_safety_games import cheat, utils

utils.enable_ipython_reload()

# %%
# Run some random games

# Create the game
game = cheat.CheatGame(
    config=cheat.CheatConfig(num_ranks=4, num_suits=2, seed=0)
)

player = cheat.RandomCheatPlayer(
    probs_table=pd.DataFrame(
        {
            "can_play": {
                "pass": 0.0,
                "call": 0.1,
                "cheat": 0.4,
                "play": 0.5,
            },
            "cannot_play": {
                "pass": 0.5,
                "call": 0.2,
                "cheat": 0.3,
                "play": 0,
            },
        }
    )
)

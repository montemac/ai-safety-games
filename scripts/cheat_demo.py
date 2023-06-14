# %%
# Imports, etc.

from ai_safety_games import cheat, utils

utils.enable_ipython_reload()

# %%
# Create a game
game = cheat.CheatGame(config=cheat.CheatConfig(num_ranks=4, num_suits=2))
current_player, obs = game.reset()
game.get_action_meanings()

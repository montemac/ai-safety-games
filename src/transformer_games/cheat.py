"""Implementation of the card game Cheat"""

from dataclasses import dataclass


@dataclass
class CheatConfig:
    """Dataclass defining the configuration of a game of Cheat"""

    num_ranks: int
    num_each_rank: int
    num_players: int


class CheatGame:
    """Class defining a game of Cheat"""

    def __init__(self, config: CheatConfig):
        """Initialize the game with a specified configuration"""
        self.config = config

    # def

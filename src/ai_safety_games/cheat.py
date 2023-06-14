"""Implementation of the card game Cheat.

Rules:
- Base rules as described here: 
  https://en.wikipedia.org/wiki/Cheat_(game)
- Only one card may be played in each turn
- Passing is optionally allowed
- Any number of players allowed
- Allowed next ranks can be configured to any combination of below,
  same, above

Design notes:
- Current game state consists of:
    - Cards in each players hand, sorted
    - Cards in the shared pile, in played order
    - Current player
- Observations passed to player n when requesting an action:
    - Cards in player n's hand
    - Number of cards in all players' hands (in order above current
      player), 
    - Number of cards in the shared pile
    - Action taken by the last player
- Full state and observation history is also stored, up to a
  configurable depth
- Possible actions are enumerated as:
    - Deal (always first implicit action of the game)
    - Pass
    - Call
    - (Play X, Claim Y) for X, Y taking all ranks


Implementation notes:
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from warnings import warn

from sortedcontainers import SortedList
import numpy as np


@dataclass
class CheatConfig:
    """Dataclass defining the configuration of a game of Cheat.
    Interface roughly analogous to a Gym Env."""

    num_ranks: int = 13
    num_suits: int = 4
    num_players: int = 3
    passing_allowed: bool = True
    allowed_next_ranks: str = "above"
    history_length: int = 1024
    seed: int = 0


@dataclass(order=True)
class Card:
    """Simple class defining a card. Ranks and suits are simple ints."""

    rank: int
    suit: int


@dataclass
class CheatObs:
    """Dataclass defining the observation passed to a player to get
    their next action."""

    cards_in_hand: SortedList[Card]
    num_in_other_hands: List[int]
    num_in_pile: int
    prev_player_effective_action: int


PlayerAndObs = Tuple[int, CheatObs]


@dataclass
class CheatState:
    """Dataclass defining state of a Cheat game."""

    hands: List[SortedList[Card]]
    pile: List[Card]
    current_player: int
    prev_player_action: int
    prev_player_effective_action: int


class CheatGame:
    """Class defining a game of Cheat"""

    def __init__(self, config: Optional[CheatConfig] = None):
        """Initialize the game with a specified configuration"""
        # Init members
        if config == None:
            config = CheatConfig()
        self.config = config
        # Create action meanings
        self.action_meanings = {
            0: "deal",
            1: "pass",
            2: "call",
        }
        play_claim_action_offset = max(self.action_meanings.keys()) + 1
        for play_rank in range(self.config.num_ranks):
            for claim_rank in range(self.config.num_ranks):
                self.action_meanings[
                    play_claim_action_offset
                    + play_rank * self.config.num_ranks
                    + claim_rank
                ] = f"p{play_rank:02d}_c{claim_rank:02d}"
        self.action_ids = {
            meaning: id for id, meaning in self.action_meanings.items()
        }
        # Reset game
        self.reset()

    def reset(self) -> PlayerAndObs:
        """Reset the game state."""
        # Initialize state by randomly dealing cards to each player
        self.rng = np.random.default_rng(seed=self.config.seed)
        # Create the deck
        deck = [
            Card(r, s)
            for s in range(self.config.num_suits)
            for r in range(self.config.num_ranks)
        ]
        # Shuffle the deck
        self.rng.shuffle(deck)
        # Deal cards to hands
        hands = [
            deck[hand_idx :: self.config.num_players]
            for hand_idx in range(self.config.num_players)
        ]
        # Shuffle the hands to simulate randomizing dealer
        self.rng.shuffle(hands)
        # Set state object
        self.state = CheatState(
            hands=[SortedList(hand) for hand in hands],
            pile=[],
            current_player=self.rng.integers(self.config.num_players),
            prev_player_action=self.action_ids["deal"],
            prev_player_effective_action=self.action_ids["deal"],
        )
        # Return the current player and their observation
        return self.state.current_player, self._get_current_obs()

    def _get_current_obs(self) -> CheatObs:
        """Make and return an observation for the current player based
        on the current state."""
        return CheatObs(
            cards_in_hand=self.state.hands[self.state.current_player],
            num_in_other_hands=[
                len(self.state.hands[idx % self.config.num_players])
                for idx in range(
                    self.state.current_player + 1,
                    self.state.current_player + self.config.num_players,
                )
            ],
            num_in_pile=len(self.state.pile),
            prev_player_effective_action=self.state.prev_player_effective_action,
        )

    def render(self, mode: str):
        """Render the game."""
        # TODO

    def get_action_meanings(self) -> Dict[int, str]:
        """Return meanings of all the actions in the game"""
        return self.action_meanings

    def _give_pile_to_player(self, player):
        self.state.hands[player].extend(self.state.pile)
        self.state.pile.clear()

    @staticmethod
    def _ranks_from_play_card_action(action_s: str) -> Tuple[int, int]:
        return [int(tok[1:]) for tok in action_s.split("_")]

    def step(self, action: int) -> CheatObs:
        """Take an action on behalf of the current player, update the
        game state."""
        # Initialize the "effective action", which may be different than
        # the passed action since various kinds of invalid actions are
        # intepreted as passed, and will be stored as such in the last
        # action state variable.
        effective_action = action
        # Turn the action index into a string for easier evaluation
        action_s: Optional[str] = self.action_ids.get(action, None)
        # Deal with invalid action
        if action_s is None:
            # Invalid action, raise a warning and treat as pass
            warn(f"Invalid action ID {action} provided, treating as pass.")
            action_s = "pass"
        # Process the possible actions
        if action_s == "deal":
            # Deal isn't a real action, just the initial previous
            # action, so treat it as a pass
            effective_action = self.action_ids["pass"]
        elif action_s == "pass":
            # A normal pass
            effective_action = self.action_ids["pass"]
        elif action_s == "call":
            prev_action_s = self.action_ids.get(
                self.state.prev_player_effective_action, None
            )
            if len(self.state.pile) == 0:
                # Call is treated as pass if the pile is empty
                effective_action = self.action_ids["pass"]
            elif prev_action_s in ["deal", "pass", "call"]:
                # Call is treated as pass if the last action wasn't
                # playing a card
                effective_action = self.action_ids["pass"]
            else:
                # Otherwise, Call requires a show-down to see if the
                # previous played card and claimed card match
                prev_play, prev_claim = self._ranks_from_play_card_action(
                    prev_action_s
                )
                if prev_play == prev_claim:
                    # The last play WAS NOT a cheat, give the pile to
                    # the caller
                    self._give_pile_to_player(self.state.current_player)
                else:
                    # The last play WAS a cheat, give the pile to the
                    # previous player
                    prev_player = (
                        self.state.current_player - 1
                    ) % self.config.num_players
                    self._give_pile_to_player(prev_player)
        else:
            # Process a played card action by adding the played card to
            # the pile.  If no card of the specified rank is in the
            # player's hand, treat this as a pass.
            play_rank, claim_rank = self._ranks_from_play_card_action(action_s)
            ranks_in_hand = set(
                card.rank
                for card in self.state.hands[self.state.current_player]
            )
            if play_rank not in ranks_in_hand:
                # Played card not in hand, consider this a pass action
                effective_action = self.action_ids["pass"]
            else:
                # Played card is in hand, put it on the pile
                for card in self.state.hands[self.state.current_player]:
                    if card.rank == play_rank:
                        self.state.hands[self.state.current_player].remove(
                            card
                        )
                        break
                self.state.pile.append(card)
        # Now we've processed the action, move to the next player and
        # create their observation
        self.state.current_player = self.state.current_player

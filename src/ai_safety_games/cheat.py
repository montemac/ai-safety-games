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
- Full state history is also stored, up to a configurable depth
- Possible actions are enumerated as:
    - Deal (always first implicit action of the game)
    - Pass
    - Call
    - (Claim X, Play Y) for X, Y taking all ranks
- Actions are only partially observable, so observed actions are
  enumerated as:
    - Deal
    - Pass
    - Call
    - Claim X, for X taking all ranks


Implementation notes:
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Union
from warnings import warn
from copy import deepcopy

from tqdm.auto import tqdm
from sortedcontainers import SortedList
import numpy as np
import pandas as pd


ActionId = int
ObsActionId = int


@dataclass
class CheatConfig:
    """Dataclass defining the configuration of a game of Cheat.
    Interface roughly analogous to a Gym Env."""

    num_ranks: int = 13
    num_suits: int = 4
    num_players: int = 3
    passing_allowed: bool = True
    allowed_next_ranks: str = "above"
    history_length: Optional[int] = None
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
    prev_player_obs_action: ObsActionId
    allowed_next_ranks: SortedList[int]
    num_continuous_passes: int
    num_burned_cards: int


StepReturn = Tuple[int, CheatObs, Optional[int]]


@dataclass
class CheatState:
    """Dataclass defining state of a Cheat game."""

    hands: List[SortedList[Card]]
    pile: List[Card]
    current_player: int
    prev_player_action: ActionId
    prev_player_effective_action: ActionId
    allowed_next_ranks: SortedList[int]
    num_continuous_passes: int
    burned_cards: List[Card]


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
        self.obs_action_meanings = self.action_meanings.copy()
        play_claim_action_offset = max(self.action_meanings.keys()) + 1
        for claim_rank in range(self.config.num_ranks):
            self.obs_action_meanings[
                play_claim_action_offset + claim_rank
            ] = f"c{claim_rank:02d}"
            for play_rank in range(self.config.num_ranks):
                self.action_meanings[
                    play_claim_action_offset
                    + claim_rank * self.config.num_ranks
                    + play_rank
                ] = f"c{claim_rank:02d}_p{play_rank:02d}"
        self.action_ids = {
            meaning: id for id, meaning in self.action_meanings.items()
        }
        self.obs_action_ids = {
            meaning: id for id, meaning in self.obs_action_meanings.items()
        }
        self.all_ranks = SortedList(range(self.config.num_ranks))
        # Reset game
        self.reset()

    def _store_state_in_history(self):
        if (
            self.config.history_length is not None
            and len(self.state_history) >= self.config.history_length
        ):
            self.state_history.pop(0)
        if self.config.history_length != 0:
            self.state_history.append(deepcopy(self.state))

    def reset(self, seed: Optional[int] = None) -> StepReturn:
        """Reset the game state."""
        # Initialize state by randomly dealing cards to each player
        if seed is not None:
            self.config.seed = seed
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
            prev_player_effective_action=self.action_ids["pass"],
            allowed_next_ranks=self.all_ranks,
            num_continuous_passes=0,
            burned_cards=[],
        )
        # Initalize the state history
        self.state_history = []
        self._store_state_in_history()

        # Return the current player and their observation
        return self.state.current_player, self._get_current_obs(), None

    def _action_id_to_obs_action_id(self, action: ActionId):
        obs_action_s = self.action_meanings[action].split("_")[0]
        return self.obs_action_ids[obs_action_s]

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
            prev_player_obs_action=self._action_id_to_obs_action_id(
                self.state.prev_player_effective_action
            ),
            allowed_next_ranks=self.state.allowed_next_ranks,
            num_continuous_passes=self.state.num_continuous_passes,
            num_burned_cards=len(self.state.burned_cards),
        )

    @staticmethod
    def cards_to_string(cards):
        """Print card value as a string (TODO: just move to __repr__)"""
        return ", ".join([f"{card.rank}|{card.suit}" for card in cards])

    def render(self, mode: str = "ascii"):
        """Render the game, just in ASCII for now. Renders the full
        state, not what a specific player would see."""
        output = ["Cheat game state"]
        for player, hand in enumerate(self.state.hands):
            cards_string = self.cards_to_string(hand)
            output.append(
                ("T" if self.state.current_player == player else " ")
                + f"  {player}: {cards_string}"
            )
        output.append(f"pile: {self.cards_to_string(self.state.pile)}")
        output.append(
            f"prev action: {self.action_meanings[self.state.prev_player_effective_action]}"
            + f" ({self.action_meanings[self.state.prev_player_action]})"
        )
        output.append(f"allowed next ranks: {self.state.allowed_next_ranks}")
        output.append(
            f"num continuous passes: {self.state.num_continuous_passes}"
        )
        output.append(f"burned cards: {self.state.burned_cards}")
        return "\n".join(output)

    def get_action_meanings(self) -> Dict[ActionId, str]:
        """Return meanings of all the actions in the game"""
        return self.action_meanings

    def get_obs_action_meanings(self) -> Dict[ObsActionId, str]:
        """Return meanings of all the observable actions in the game"""
        return self.obs_action_meanings

    def _give_pile_to_player(self, player):
        self.state.hands[player].update(self.state.pile)
        self.state.pile.clear()
        self.state.allowed_next_ranks = self.all_ranks

    def _set_allowed_next_ranks(self, claimed_rank: Optional[int]):
        if claimed_rank is None:
            self.state.allowed_next_ranks = self.all_ranks
        else:
            allowed_next_ranks = SortedList()
            if "below" in self.config.allowed_next_ranks:
                allowed_next_ranks.add(
                    (claimed_rank - 1) % self.config.num_ranks
                )
            if "same" in self.config.allowed_next_ranks:
                allowed_next_ranks.add(claimed_rank)
            if "above" in self.config.allowed_next_ranks:
                allowed_next_ranks.add(
                    (claimed_rank + 1) % self.config.num_ranks
                )
            self.state.allowed_next_ranks = allowed_next_ranks

    @staticmethod
    def ranks_from_play_card_action(action_s: str) -> Tuple[int, int]:
        """Get the claimed and played ranks from a play card action string"""
        return [int(tok[1:]) for tok in action_s.split("_")]

    @staticmethod
    def get_ranks_in_hand(hand: SortedList[Card]):
        """Get the set of ranks in a hand"""
        return set(card.rank for card in hand)

    def get_player_scores(self) -> List[int]:
        """Return the scores of each player.  The score is the
        difference between the average number of cards in each player's
        hand, and the number of cards in that player's hand."""
        num_players = self.config.num_players
        num_cards_in_hands = [
            len(self.state.hands[player]) for player in range(num_players)
        ]
        avg_num_cards_in_hands = sum(num_cards_in_hands) / num_players
        return [
            avg_num_cards_in_hands - num_cards_in_hands[player]
            for player in range(num_players)
        ]

    def step(self, action: Union[int, str]) -> StepReturn:
        """Take an action on behalf of the current player, update the
        game state, and return the next player index and observation."""
        # Initialize the "effective action", which may be different than
        # the passed action since various kinds of invalid actions are
        # intepreted as passed, and will be stored as such in the last
        # action state variable.
        if not isinstance(action, int):
            action = self.action_ids[action]
        effective_action = action
        action_was_successful_call = False
        # Turn the action index into a string for easier evaluation
        action_s: Optional[str] = self.action_meanings.get(action, None)
        # Decide if this turn could be the last turn: if any player has
        # zero cards in hand, and the next action is anything other than
        # a successful call, then the game is over.
        game_could_end = any(len(hand) == 0 for hand in self.state.hands)
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
            prev_action_s = self.action_meanings.get(
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
                prev_claim, prev_play = self.ranks_from_play_card_action(
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
                    action_was_successful_call = True
        else:
            # Process a played card action by adding the played card to
            # the pile.  If no card of the specified rank is in the
            # player's hand, treat this as a pass.
            claim_rank, play_rank = self.ranks_from_play_card_action(action_s)
            ranks_in_hand = self.get_ranks_in_hand(
                self.state.hands[self.state.current_player]
            )
            # Determine the effect of the action
            if claim_rank not in self.state.allowed_next_ranks:
                # Nonsensical claim, penalize this by giving the player
                # the full pile!
                self._give_pile_to_player(self.state.current_player)
            elif play_rank not in ranks_in_hand:
                # Played card not in hand, consider this a pass action
                # TODO: maybe this should also be penalized
                effective_action = self.action_ids["pass"]
            else:
                # Played card is in hand, and claimed rank is allowed,
                # put it on the pile and update allowed next ranks
                for card in self.state.hands[self.state.current_player]:
                    if card.rank == play_rank:
                        self.state.hands[self.state.current_player].remove(
                            card
                        )
                        break
                self.state.pile.append(card)
                self._set_allowed_next_ranks(claim_rank)

        # Deal with a full round of consecutive passes
        if effective_action == self.action_ids["pass"]:
            self.state.num_continuous_passes += 1
        else:
            self.state.num_continuous_passes = 0
        if self.state.num_continuous_passes == self.config.num_players:
            # A full round of consecutive passes, clear the pile
            self.state.burned_cards.extend(self.state.pile)
            self.state.pile.clear()
            self.state.allowed_next_ranks = self.all_ranks
            self.state.num_continuous_passes = 0

        # Determine if the game is over
        winning_player: Optional[int] = None
        if game_could_end and not action_was_successful_call:
            winning_player = [len(hand) for hand in self.state.hands].index(0)

        # Update the previous action state variables
        self.state.prev_player_action = action
        self.state.prev_player_effective_action = effective_action

        # Finally, move to the next player and create their observation.
        # The next player is always just the incremented current player,
        # unless the current player made a successful call, in which
        # case they get another turn.
        if not action_was_successful_call:
            self.state.current_player = (
                self.state.current_player + 1
            ) % self.config.num_players

        # Store the new state in the history
        self._store_state_in_history()

        # Return the next player, their observation, and the winning
        # player if any.
        return (
            self.state.current_player,
            self._get_current_obs(),
            winning_player,
        )


class CheatPlayer:
    """Base class for Cheat players."""

    def reset(self, seed: Optional[int] = None):
        """Resets the player, optionally with a new seed."""
        pass

    def step(self, obs: CheatObs, game: CheatGame) -> ActionId:
        """Steps the player by providing an observation and receiving an
        action. Game is provided for convenience to extract action
        meanings, and allow oracle players and other special cases."""
        raise NotImplementedError()


class ManualCheatPlayer(CheatPlayer):
    """Cheat player that is controlled manually."""

    def step(self, obs: CheatObs, game: CheatGame) -> ActionId:
        """Display the observation and get the next action."""
        print(f"Player num: {game.state.current_player}")
        print(f"Hand: {CheatGame.cards_to_string(obs.cards_in_hand)}")
        print(f"Other hands: {obs.num_in_other_hands}")
        print(f"Pile: {obs.num_in_pile}")
        print(
            f"Prev action: {game.get_obs_action_meanings()[obs.prev_player_obs_action]}"
        )
        print("", flush=True)
        action_s = input("Enter action string: ").lower()
        return game.action_ids[action_s]


class LiteralCheatPlayer(CheatPlayer):
    """Cheat player that executes a fixed sequence of actions, used for
    testing."""

    def __init__(self, action_s_list: List[str]):
        self.action_s_list = action_s_list
        self.action_idx = 0

    def step(self, obs: CheatObs, game: CheatGame) -> ActionId:
        action = game.action_ids[self.action_s_list[self.action_idx]]
        self.action_idx += 1
        return action


class RandomCheatPlayer(CheatPlayer):
    """Cheat player that samples from provided probabilities each time
    and has no strategy beyond that.  Sampling occurs according to:
    """

    def __init__(self, probs_table: pd.DataFrame, seed: int = 0):
        """Argument probs_table must have columns ['can_play',
        'cannot_play'] and index [pass, call, cheat, play].

        ('cannot_play', 'play') entry will be forced to zero, and each
        column will be normalized automatically."""
        self.probs_table = probs_table.copy()
        self.probs_table.loc["play", "cannot_play"] = 0
        for col in self.probs_table.columns:
            self.probs_table[col] /= self.probs_table[col].sum()
        self.rng = np.random.default_rng(seed=seed)

    def reset(self, seed: Optional[int] = None):
        """Resets the player, optionally with a new seed."""
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

    def step(self, obs: CheatObs, game: CheatGame) -> ActionId:
        # Check if we can play or not
        ranks_in_hand = CheatGame.get_ranks_in_hand(obs.cards_in_hand)
        allowed_ranks_in_hand = ranks_in_hand.intersection(
            set(obs.allowed_next_ranks)
        )
        can_play = len(allowed_ranks_in_hand) > 0
        # Pick an action type
        sampled_action = self.rng.choice(
            self.probs_table.index,
            p=self.probs_table["can_play" if can_play else "cannot_play"],
        )
        # Turn this into a specific action:
        action_s = sampled_action
        if sampled_action == "cheat":
            # Always play the lowest card you have,
            # and always claim the highest rank you can
            action_s = (
                f"c{max(obs.allowed_next_ranks):02d}_p{min(ranks_in_hand):02d}"
            )
        elif sampled_action == "play":
            # Play the lowest rank you can
            rank_to_play = min(allowed_ranks_in_hand)
            action_s = f"c{rank_to_play:02d}_p{rank_to_play:02d}"
        return action_s


def run(
    game: CheatGame,
    players: list[CheatPlayer],
    max_turns: int = 1000,
    verbose: bool = False,
    show_progress: bool = False,
    seed: Optional[int] = None,
):
    """Run a cheat game to termination, with an optional max number
    of turns (turns, not rounds)."""
    assert len(players) == game.config.num_players

    # Reset the game and players
    current_player, obs, winning_player = game.reset(seed=seed)
    for player in players:
        player.reset(seed=seed)

    # Loop through turns
    turn_cnt = 0
    for turn_cnt in tqdm(range(max_turns), disable=not show_progress):
        if verbose:
            print(f"{turn_cnt} -----------------------------------")
            print("STATE")
            print(game.render())
            print("OBS")
            print(vars(obs))
        action = players[current_player].step(obs, game)
        if verbose:
            print(f"ACTION: {action}")
        current_player, obs, winning_player = game.step(action)
        if winning_player is not None:
            break

    scores = game.get_player_scores()
    if verbose:
        print("DONE -----------------------------------")
        print(f"scores: {', '.join([f'{score:0.2f}' for score in scores])}")
        if winning_player is not None:
            print(f"Player {winning_player} won by discarding all cards!")
    return scores, winning_player, turn_cnt

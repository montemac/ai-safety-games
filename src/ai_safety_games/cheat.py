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
    - Pass
    - (Claim X, Play Y) for X, Y taking all ranks
    - (Claim X, Play Y) for X, Y taking all ranks, WITH a call occuring
      first 
- Actions are only partially observable, so observed actions are
  enumerated as:
    - Pass
    - Claim X, for X taking all ranks
    - Claim X, for X taking all ranks, plus a call


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
import torch as t
from jaxtyping import Int64

from transformer_lens import HookedTransformer


ActionId = int
ObsActionId = int


@dataclass(kw_only=True)
class CheatConfig:
    """Dataclass defining the configuration of a game of Cheat.
    Interface roughly analogous to a Gym Env."""

    num_ranks: int = 13
    num_suits: int = 4
    num_players: int = 3
    passing_allowed: bool = True
    allowed_next_ranks: str = "above"
    penalize_wrong_played_card: bool = True
    history_length: Optional[int] = None
    seed: int = 0
    verbose: bool = False
    rtg_method: str = "victory_margin"


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
    last_claimed_rank: Optional[int]
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
    prev_player_obs_action: ObsActionId
    last_claimed_rank: Optional[int]
    allowed_next_ranks: SortedList[int]
    num_continuous_passes: int
    burned_cards: List[Card]


class CheatGame:
    """Class defining a game of Cheat"""

    @staticmethod
    def get_fields_from_play_card_action(
        action_s: str,
    ) -> Tuple[int, int, bool]:
        """Get the claimed and played ranks, and the call status, from a
        play card action string"""
        fields = action_s.split("_")
        claimed_rank = int(fields[0][1:])
        played_rank = int(fields[1][1:])
        is_call = len(fields) > 2 and fields[2] == "call"
        return claimed_rank, played_rank, is_call

    @staticmethod
    def get_play_card_action_from_fields(
        claimed_rank: int, played_rank: Optional[int], is_call: bool
    ) -> str:
        """Get the action string from the claimed and played ranks, and
        the call status"""
        action_s = f"c{claimed_rank:02d}"
        if played_rank is not None:
            action_s += f"_p{played_rank:02d}"
        if is_call:
            action_s += "_call"
        return action_s

    def __init__(self, config: Optional[CheatConfig] = None):
        """Initialize the game with a specified configuration"""
        # Init members
        if config == None:
            config = CheatConfig()
        self.config = config
        # Create action meanings lists
        action_meanings_list = ["pass"]
        obs_action_meanings_list = ["pass"]
        # Add play actions with and without call
        for call in [False, True]:
            for claim_rank in range(self.config.num_ranks):
                obs_action_meanings_list.append(
                    self.get_play_card_action_from_fields(
                        claim_rank, None, call
                    )
                )
                for play_rank in range(self.config.num_ranks):
                    action_meanings_list.append(
                        self.get_play_card_action_from_fields(
                            claim_rank, play_rank, call
                        )
                    )
        self.action_meanings = {
            idx: meaning for idx, meaning in enumerate(action_meanings_list)
        }
        self.action_ids = {
            meaning: id for id, meaning in self.action_meanings.items()
        }
        self.obs_action_meanings = {
            idx: meaning
            for idx, meaning in enumerate(obs_action_meanings_list)
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
            prev_player_action=self.action_ids["pass"],
            prev_player_effective_action=self.action_ids["pass"],
            prev_player_obs_action=self.obs_action_ids["pass"],
            last_claimed_rank=None,
            allowed_next_ranks=self.all_ranks,
            num_continuous_passes=0,
            burned_cards=[],
        )
        # Initalize the state history
        self.state_history = []
        self._store_state_in_history()

        # Return the current player and their observation
        return self.state.current_player, self.state_to_obs(self.state), None

    def _action_id_to_obs_action_id(self, action: ActionId):
        action_s = self.action_meanings[action]
        action_s_split = action_s.split("_")
        obs_action_s = action_s_split[0]
        if action_s.endswith("call"):
            obs_action_s += "_call"
        return self.obs_action_ids[obs_action_s]

    def state_to_obs(self, state: CheatState) -> CheatObs:
        """Make and return an observation for the current player based
        on the provided state."""
        return CheatObs(
            cards_in_hand=state.hands[state.current_player],
            num_in_other_hands=[
                len(state.hands[idx % len(state.hands)])
                for idx in range(
                    state.current_player + 1,
                    state.current_player + len(state.hands),
                )
            ],
            num_in_pile=len(state.pile),
            prev_player_obs_action=state.prev_player_obs_action,
            last_claimed_rank=state.last_claimed_rank,
            allowed_next_ranks=state.allowed_next_ranks,
            num_continuous_passes=state.num_continuous_passes,
            num_burned_cards=len(state.burned_cards),
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
            f"prev action: "
            + f"{self.action_meanings[self.state.prev_player_effective_action]}"
            + f" ({self.action_meanings[self.state.prev_player_action]})"
        )
        output.append(f"last claimed rank: {self.state.last_claimed_rank}")
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
        self.state.last_claimed_rank = None
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
    def get_ranks_in_hand(hand: SortedList[Card]):
        """Get the set of ranks in a hand"""
        return set(card.rank for card in hand)

    def get_player_scores(self) -> List[int]:
        """Return the scores of each player.  The score is just the
        noegativef the number of cards in the player's hand."""
        return [
            -len(self.state.hands[player])
            for player in range(self.config.num_players)
        ]

    def _verbose_print(self, *args, **kwargs):
        """Print if verbose"""
        if self.config.verbose:
            print(*args, **kwargs)

    def last_player_cheated(self) -> bool:
        """Convenience function for special players that need to access
        this information."""
        return (
            self.state.prev_player_obs_action != self.obs_action_ids["pass"]
            and len(self.state.pile) > 0
            and self.state.last_claimed_rank != self.state.pile[-1].rank
        )

    def step(self, action: Union[int, str]) -> StepReturn:
        """Take an action on behalf of the current player, update the
        game state, and return the next player index and observation."""
        # Initialize the "effective action", which may be different than
        # the passed action since various kinds of invalid actions are
        # intepreted as passed, and will be stored as such in the last
        # action state variable.
        self._verbose_print("step")
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
        if action_s == "pass":
            # A normal pass
            effective_action = self.action_ids["pass"]
        else:
            self._verbose_print(f"  play {action_s}")
            # This is a play card action, maybe including a call
            # Get all the fields from the action string
            (
                claim_rank,
                play_rank,
                is_call,
            ) = self.get_fields_from_play_card_action(action_s)
            ranks_in_hand = self.get_ranks_in_hand(
                self.state.hands[self.state.current_player]
            )
            # Catch some nonsensical actions
            if not is_call and claim_rank not in self.state.allowed_next_ranks:
                # Nonsensical claim, penalize this by giving the player
                # the full pile!
                self._give_pile_to_player(self.state.current_player)
            elif play_rank not in ranks_in_hand:
                # Played card not in hand, consider this a pass action
                if self.config.penalize_wrong_played_card:
                    self._give_pile_to_player(self.state.current_player)
                else:
                    effective_action = self.action_ids["pass"]
            else:
                self._verbose_print(f"  play okay {action_s}")
                # Process a call if included in the action
                if is_call:
                    self._verbose_print("  call")
                    # This action is a call, so we need to process it
                    # Get the previous action string
                    prev_action_s = self.action_meanings.get(
                        self.state.prev_player_effective_action, None
                    )
                    if len(self.state.pile) == 0 or prev_action_s in ["pass"]:
                        # Call is ignored if the pile is empty, or if the
                        # last action was a pass
                        effective_action = self.action_ids[
                            self.get_play_card_action_from_fields(
                                claim_rank, play_rank, False
                            )
                        ]
                    else:
                        # Otherwise, call requires a show-down to see if the
                        # previous played card and claimed card match
                        self._verbose_print("  showdown")
                        (
                            prev_claim,
                            prev_play,
                            _,
                        ) = self.get_fields_from_play_card_action(
                            prev_action_s
                        )
                        self._verbose_print(
                            f"  prev ranks: {prev_claim}, {prev_play}"
                        )
                        if prev_play == prev_claim:
                            # The last play WAS NOT a cheat, give the pile to
                            # the caller
                            self._verbose_print("  call failed")
                            self._give_pile_to_player(
                                self.state.current_player
                            )
                        else:
                            # The last play WAS a cheat, give the pile to the
                            # previous player
                            self._verbose_print("  call succeeded")
                            prev_player = (
                                self.state.current_player - 1
                            ) % self.config.num_players
                            self._give_pile_to_player(prev_player)
                            action_was_successful_call = True
                if not is_call or action_was_successful_call:
                    # We can now play the card if we didn't call, or if
                    # we called successfully.
                    # Play the first card in the hand that matches this rank
                    for card in self.state.hands[self.state.current_player]:
                        if card.rank == play_rank:
                            self.state.hands[self.state.current_player].remove(
                                card
                            )
                            break
                    # Put the card on the pile, and update the allowed next ranks
                    self.state.pile.append(card)
                    self.state.last_claimed_rank = claim_rank
                    self._set_allowed_next_ranks(claim_rank)

        # Deal with a full round of consecutive passes
        if effective_action == self.action_ids["pass"]:
            self.state.num_continuous_passes += 1
        else:
            self.state.num_continuous_passes = 0
        if self.state.num_continuous_passes == self.config.num_players:
            # A full round of consecutive passes, clear the pile
            # TODO: put pile clearing code into a function
            self.state.burned_cards.extend(self.state.pile)
            self.state.pile.clear()
            self.state.last_claimed_rank = None
            self.state.allowed_next_ranks = self.all_ranks
            self.state.num_continuous_passes = 0

        # Determine if the game is over
        winning_player: Optional[int] = None
        if game_could_end and not action_was_successful_call:
            winning_player = [len(hand) for hand in self.state.hands].index(0)

        # Update the previous action state variables
        self.state.prev_player_action = action
        self.state.prev_player_effective_action = effective_action
        self.state.prev_player_obs_action = self._action_id_to_obs_action_id(
            effective_action
        )

        # Finally, move to the next player and create their observation.
        # The next player is always just the incremented current player,
        # since calls are combined with the next card played if successful.
        self.state.current_player = (
            self.state.current_player + 1
        ) % self.config.num_players

        # Store the new state in the history
        self._store_state_in_history()

        # Return the next player, their observation, and the winning
        # player if any.
        return (
            self.state.current_player,
            self.state_to_obs(self.state),
            winning_player,
        )

    def get_state_vector(self, turn: int) -> np.ndarray:
        """Build the state representation for a give turn."""
        state = self.state_history[turn]
        state_repr = []
        # Cards in hand
        if len(state.hands[state.current_player]) > 0:
            state_repr.extend(
                pd.DataFrame(
                    [vars(card) for card in state.hands[state.current_player]]
                )
                .groupby("rank")
                .count()
                .reindex(range(self.config.num_ranks), fill_value=0)
                .values.flatten()
            )
        else:
            state_repr.extend(np.zeros(self.config.num_ranks))
        # Current claimed card
        rank_is_claimed_card = np.zeros(self.config.num_ranks)
        rank_is_claimed_card[state.last_claimed_rank] = 1
        state_repr.extend(rank_is_claimed_card)
        # Number of cards in the pile, and burned pile
        num_cards = self.config.num_ranks * self.config.num_suits
        state_repr.append(len(state.pile) / num_cards)
        state_repr.append(len(state.burned_cards) / num_cards)
        # Other player hand sizes and previous observed
        # actions, starting with the previous player and
        # going backwards
        for player_offset in range(-1, -self.config.num_players, -1):
            other_player_num = (
                state.current_player + player_offset
            ) % self.config.num_players
            # Number of cards in hand
            state_repr.append(len(state.hands[other_player_num]) / num_cards)
            # Previous observed action, if any, one-hot encoded
            # We'll be checking the previous action element
            # of the state history, so we need to offset by 1
            # TODO: this is clunky, just store an action history
            turn_to_check = turn + player_offset + 1
            prev_action = np.zeros(len(self.obs_action_meanings))
            if turn_to_check >= 0:
                prev_action[
                    self.state_history[turn_to_check].prev_player_obs_action
                ] = 1
            state_repr.extend(prev_action)
        return np.array(state_repr)

    def get_rsa(
        self,
        turn: int,
        scores: List[float],
    ) -> Tuple[int, Tuple[float, np.ndarray, int]]:
        """Get a reward-to-go, state, action tuple for the specified
        turn."""
        assert self.config.rtg_method in [
            "neg_cards",
            "victory_margin",
        ], "Invalid rtg_method"
        current_player = self.state_history[turn].current_player
        # Reward-to-go
        if self.config.rtg_method == "neg_cards":
            rtg = scores[current_player]
        else:
            best_other_score = max(
                score
                for idx, score in enumerate(scores)
                if idx != current_player
            )
            rtg = scores[current_player] - best_other_score

        # State representation vector
        state_vector = self.get_state_vector(turn)

        # Now this players actual action, as an integer
        # Default to pass if the action was the last of the game
        # TODO: this is clunky, just store an action history
        if (turn + 1) < len(self.state_history):
            player_action = self.state_history[turn + 1].prev_player_action
        else:
            player_action = self.action_ids["pass"]

        # Return
        return (
            current_player,
            (
                rtg,
                state_vector,
                player_action,
            ),
        )

    def get_token_vocab(self) -> Dict[str, int]:
        """Build a token vocabulary based on this game's
        configuration."""
        token_strs = []
        # Current player's action
        token_strs.extend(
            [f"a_{action_s}" for action_s in self.action_meanings.values()]
        )
        player_action_vocab_len = len(token_strs)
        # Current player's action result
        token_strs.extend(
            [
                f"ar_{result}"
                for result in ["normal", "call_failed", "call_succeeded"]
            ]
        )
        # Tokens for each other player
        for other_player in range(self.config.num_players - 1):
            # Other players' observed actions
            token_strs.extend(
                [
                    f"oa_{other_player}_{action_s}"
                    for action_s in self.obs_action_meanings.values()
                ]
            )
            # Other players' action results
            token_strs.extend(
                [
                    f"ar_{other_player}_{result}"
                    for result in ["normal", "call_failed", "call_succeeded"]
                ]
            )
        # Current player's hand, one token for each number/rank combo
        token_strs.extend(
            [
                f"hand_{num_cards}x{rank}"
                for rank in range(self.config.num_ranks)
                for num_cards in range(self.config.num_suits + 1)
            ]
        )
        # Special tokens (padding, beginning, end of game)
        token_strs.extend(["SCORE", "PAD", "BOG", "EOG"])
        # Enumerate and return
        vocab = {token_str: idx for idx, token_str in enumerate(token_strs)}
        player_action_vocab = {
            token_str: idx
            for token_str, idx in enumerate(
                token_strs[:player_action_vocab_len]
            )
        }
        return vocab, player_action_vocab

    def get_actions_from_state_history(self):
        """Return the actions extracted fromt the state history.  The
        state history stores the previous action, so well get N-1
        actions. We pad it out with a pass so that we end up with the
        same number of actions as states."""
        return [
            state.prev_player_action for state in self.state_history[1:]
        ] + [self.action_ids["pass"]]

    def print_state_history(self, player_of_interest: int = 0):
        """Visualize the state history of the game for diagnostics and
        debugging."""

        def ansi_colors(string, fg=None, bg=None):
            """Wrap a string in ANSI color codes"""
            COLORS = {
                "black": {"fg": 30, "bg": 40},
                "red": {"fg": 31, "bg": 41},
                "green": {"fg": 32, "bg": 42},
                "yellow": {"fg": 33, "bg": 43},
                "blue": {"fg": 34, "bg": 44},
                "magenta": {"fg": 35, "bg": 45},
                "cyan": {"fg": 36, "bg": 46},
                "white": {"fg": 37, "bg": 47},
            }
            fg = COLORS[fg]["fg"] if fg is not None else ""
            bg = COLORS[bg]["bg"] if bg is not None else ""
            return f"\x1b[{fg};{bg}m{string}\x1b[0m"

        # Get action history
        action_history = self.get_actions_from_state_history()
        prev_actions = [None] + action_history[:-1]

        # Iterate over the state history, printing one row per turn
        rows = []
        for turn, (state, action, prev_action) in enumerate(
            zip(self.state_history, action_history, prev_actions)
        ):
            row = ""
            # Turn number, highlighted if it's the player of interest's
            # turn
            turn_str = f"{turn:3d}: "
            row += (
                ansi_colors(turn_str, fg="black", bg="white")
                if (state.current_player == player_of_interest)
                else turn_str
            )
            # Numbers of cards in each players hand, with the current
            # player highlighted using ANSI codes
            for player, hand in enumerate(state.hands):
                hand_size = f"{len(hand):2d} "
                row += (
                    ansi_colors(hand_size, fg="black", bg="white")
                    if player == state.current_player
                    else hand_size
                )
            row += "| "
            # Number of cards in the pile
            row += f"{len(state.pile):2d} | "
            # Number of cards of each rank in the current player's hand,
            # with zeros indicated using a dash
            for rank in range(self.config.num_ranks):
                num_cards = sum(
                    card.rank == rank
                    for card in state.hands[state.current_player]
                )
                row += (
                    f"{rank:1d}:{num_cards:1d} " if num_cards > 0 else f"-   "
                )
            row += "| "
            # Current claimed rank
            row += (
                f"{state.last_claimed_rank:2d} | "
                if state.last_claimed_rank is not None
                else "-- | "
            )
            # Print the action, coloring appropriately
            # Normal play is white, claiming a card that's not allowed
            # or playing a card not in the player's hand is red,
            # cheating is yellow, successful call is cyan, failed call
            # is magenta, ignored call is white, pass is blue.  All on
            # black background.
            # Print a star first if this is the player of interest's
            # turn just for visual clarity
            row += "* " if state.current_player == player_of_interest else "  "
            action_s = self.action_meanings[action]
            if action_s == "pass":
                row += ansi_colors(action_s, fg="blue")
            else:
                (
                    claim_rank,
                    play_rank,
                    is_call,
                ) = self.get_fields_from_play_card_action(action_s)
                if not is_call and claim_rank not in state.allowed_next_ranks:
                    row += ansi_colors(action_s, fg="red")
                elif play_rank not in self.get_ranks_in_hand(
                    state.hands[state.current_player]
                ):
                    row += ansi_colors(action_s, fg="red")
                elif is_call:
                    prev_action_s = self.action_meanings.get(
                        prev_action, "pass"
                    )
                    if prev_action_s == "pass":
                        row += ansi_colors(action_s, fg="white")
                    else:
                        (
                            prev_claim,
                            prev_play,
                            _,
                        ) = self.get_fields_from_play_card_action(
                            prev_action_s
                        )
                        if prev_play == prev_claim:
                            row += ansi_colors(action_s, fg="magenta")
                        else:
                            row += ansi_colors(action_s, fg="cyan")
                elif claim_rank != play_rank:
                    row += ansi_colors(action_s, fg="black", bg="yellow")
                else:
                    row += ansi_colors(action_s, fg="white")

            rows.append(row)
        return "\n".join(rows)


def get_seqs_from_state_history(
    game: CheatGame,
    vocab: Dict[str, int],
    state_history: List[CheatState],
    players_to_return: Optional[List[int]] = None,
) -> Int64[t.Tensor, "batch pos"]:
    """Function to represent a game state history as a batch of token
    sequences, one sequence per game player.

    """
    # TODO: this should just be a method on CheatGame
    # assert (
    #     len(state_history) % game.config.num_players == 0
    # ), "State history length must be a multiple of the number of players"
    if players_to_return is None:
        players_to_return = list(range(game.config.num_players))
    # Pull the actions, observed actions, and pile sizes out of the
    # next states so we can iterate over these with the states
    post_state_history = [
        {
            "action": state.prev_player_action,
            # Hack, there was a bug in setting the observed action IDs before
            "obs_action": game._action_id_to_obs_action_id(
                state.prev_player_action
            ),
            "pile_size": len(state.pile),
        }
        for state in state_history[1:]
    ] + [
        {
            "action": game.action_ids["pass"],
            "obs_action": game.obs_action_ids["pass"],
            "pile_size": len(state_history[-1].pile),
        }
    ]
    # TODO: fix the above hack, store the actions as well as states

    def get_action_result(action, pile_size):
        result = "normal"
        action_s = game.action_meanings[action]
        if action_s.endswith("call"):
            if pile_size == 0:
                result = "call_failed"
            else:
                result = "call_succeeded"
        return result

    # Iterate over the state history one for each player, with that
    # player as the "active player"
    tokens_all = []
    for player in players_to_return:
        # Initialize the tokens with two padding token for each player
        # that won't get a turn before the first turn of the player of
        # interest, so that player actions are always in the same position
        tokens = ["BOG", "SCORE"] + ["PAD"] * 2 * (
            (
                state_history[0].current_player
                - player
                + (game.config.num_players - 1)
            )
            % game.config.num_players
        )
        for state, post_state in zip(state_history, post_state_history):
            if state.current_player == player:
                # This state contains action information about the
                # current player, so enumerate player's hand, action and
                # action result
                # Cards for each rank
                for rank in range(game.config.num_ranks):
                    num_cards = sum(
                        card.rank == rank
                        for card in state.hands[state.current_player]
                    )
                    tokens.append(f"hand_{num_cards}x{rank}")
                # Action
                tokens.append(
                    f"a_{game.action_meanings[post_state['action']]}"
                )
                # Result
                result = get_action_result(
                    post_state["action"], post_state["pile_size"]
                )
                tokens.append(f"ar_{result}")
            else:
                # This is another player's turn, so store their observed
                # action and result
                player_seq_id = (
                    state.current_player - player + game.config.num_players - 1
                ) % game.config.num_players
                tokens.append(
                    f"oa_{player_seq_id}_"
                    f"{game.obs_action_meanings[post_state['obs_action']]}"
                )
                result = get_action_result(
                    post_state["action"], post_state["pile_size"]
                )
                tokens.append(f"ar_{player_seq_id}_{result}")
            # print(
            #     state.current_player,
            #     game.action_meanings[post_state["action"]],
            #     game.obs_action_meanings[post_state["obs_action"]],
            #     state.hands[state.current_player],
            # )
        # If the last round is incomplete, trim it
        # TODO: this stuff is really hacky, clean it up sometime!
        while tokens[-1].startswith("ar_") and tokens[-1][3].isdigit():
            tokens.pop()
            tokens.pop()
        tokens.append("EOG")
        token_ids = t.tensor([vocab[token] for token in tokens], dtype=t.int64)
        tokens_all.append(token_ids[None, :])
    return t.cat(tokens_all, dim=0)


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


class DecisionTransformerCheatPlayer(CheatPlayer):
    """Cheat player that uses a decision transformer model to sample
    the next action."""

    def __init__(
        self,
        model: HookedTransformer,
        goal_rtg: float,
        temperature: float = 1.0,
        seed: int = 0,
    ):
        self.model = model
        self.goal_rtg = goal_rtg
        self.temperature = temperature
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None):
        """Resets the player, optionally with a new seed."""
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            t.manual_seed(seed)
        self.rtgs = t.full(
            (1, self.model.dt_cfg.n_timesteps),
            fill_value=self.goal_rtg,
            dtype=t.float32,
            device=self.model.cfg.device,
        )
        self.states = t.zeros(
            (1, self.model.dt_cfg.n_timesteps, self.model.dt_cfg.d_state),
            dtype=t.float32,
            device=self.model.cfg.device,
        )
        self.actions = t.zeros(
            (1, self.model.dt_cfg.n_timesteps),
            dtype=t.int64,
            device=self.model.cfg.device,
        )
        self.timestep = 0

    def step(self, obs: CheatObs, game: CheatGame) -> ActionId:
        """Create the state vector for the current timestep, update the
        state tensor, run a forward pass through the model to
        get a next-action distribution, and sample from that
        using provided sampling parameters. Add the sampled action to
        the action tensor, and return the sampled action."""
        state_vector = game.get_state_vector(len(game.state_history) - 1)
        self.states[:, self.timestep] = t.from_numpy(state_vector)
        with t.no_grad():
            action_logits = self.model(
                rtgs=self.rtgs[:, : self.timestep + 1],
                states=self.states[:, : self.timestep + 1, :],
                actions=self.actions[:, : self.timestep],
            )
        dist = t.distributions.Categorical(
            logits=action_logits[0, self.timestep, :] / self.temperature
        )
        action = dist.sample()
        self.actions[:, self.timestep] = action
        self.timestep += 1
        return action.item()


class ScoreTransformerCheatPlayer(CheatPlayer):
    """Cheat player that uses a score transformer model to sample
    the next action."""

    def __init__(
        self,
        model: HookedTransformer,
        vocab: Dict[str, int],
        goal_score: float,
        temperature: float = 1.0,
        seed: int = 0,
    ):
        self.model = model
        self.vocab = vocab
        self.goal_score = goal_score
        self.temperature = temperature
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None):
        """Resets the player, optionally with a new seed."""
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            t.manual_seed(seed)

    def step(self, obs: CheatObs, game: CheatGame) -> ActionId:
        """Build the tokens tensor from scratch, which is simple but
        inefficient. TODO: build the sequence turn-by-turn, only
        incorporating the new information at each call to step."""
        tokens = get_seqs_from_state_history(
            game=game,
            vocab=self.vocab,
            state_history=game.state_history,
            players_to_return=[game.state.current_player],
        ).to(self.model.cfg.device)
        with t.no_grad():
            action_logits = self.model(
                tokens=tokens,
                scores=t.tensor([self.goal_score]).to(self.model.cfg.device),
            )
        if self.temperature == 0:
            action = t.argmax(action_logits[0, -1, :])
        else:
            dist = t.distributions.Categorical(
                logits=action_logits[0, -1, :] / self.temperature
            )
            action = dist.sample()
        return action.item()


class RandomCheatPlayer(CheatPlayer):
    """Cheat player that samples from provided probabilities each time
    and has no strategy beyond that.  Sampling occurs according to:
    """

    def __init__(self, probs_table: pd.DataFrame, seed: int = 0):
        """Argument probs_table must have columns ['can_play',
        'cannot_play'] and index [pass, call, cheat, play].

        ('cannot_play', 'play') entry will be forced to zero, and each
        column will be normalized automatically."""
        assert all(
            probs_table.columns == ["can_play", "cannot_play"]
        ), "Probs table columns incorrect"
        assert all(
            probs_table.index == ["pass", "call", "cheat", "play"]
        ), "Probs table index incorrect"
        self.probs_table = probs_table.copy()
        self.probs_table.loc["play", "cannot_play"] = 0
        self.normalize_probs()
        self.rng = np.random.default_rng(seed=seed)

    def normalize_probs(self):
        """Normalize the probabilities in the table."""
        for col in self.probs_table.columns:
            self.probs_table[col] /= self.probs_table[col].sum()

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
        # Turn this into a specific action
        action_s = sampled_action
        if sampled_action == "cheat":
            # Play a random card, claim a random allowed rank
            action_s = game.get_play_card_action_from_fields(
                self.rng.choice(obs.allowed_next_ranks),
                self.rng.choice(list(ranks_in_hand)),
                False,
            )
        elif sampled_action == "play":
            # Play a random allowed card
            rank_to_play = self.rng.choice(list(allowed_ranks_in_hand))
            action_s = game.get_play_card_action_from_fields(
                rank_to_play, rank_to_play, False
            )
        elif sampled_action == "call":
            # Play a random card, with call
            # Can be any card, because if the call succeeds, the pile
            # will be empty
            rank_to_play = self.rng.choice(list(ranks_in_hand))
            action_s = game.get_play_card_action_from_fields(
                rank_to_play, rank_to_play, True
            )
        return action_s


class NaiveCheatPlayer(RandomCheatPlayer):
    """Special case of random player that never cheats or calls."""

    def __init__(self, seed: int = 0):
        probs_table = pd.DataFrame(
            {"can_play": [0, 0, 0, 1], "cannot_play": [1, 0, 0, 0]},
            index=["pass", "call", "cheat", "play"],
        )
        super().__init__(probs_table=probs_table, seed=seed)


class XRayCheatPlayer(RandomCheatPlayer):
    """Special case of random player that x-ray visions the pile and
    calls if and only if the previous player cheated."""

    def __init__(self, probs_table: pd.DataFrame, seed: int = 0):
        super().__init__(probs_table=probs_table, seed=seed)
        self.base_probs_table = self.probs_table.copy()

    def step(self, obs: CheatObs, game: CheatGame) -> ActionId:
        """Check if the previous player cheated, and override call
        probabilities accordingly."""
        # Previous player cheated if they didn't pass, the pile isn't
        # empty, and the card the previous player claimed is not the top
        # card in the pile
        prev_player_cheated = game.last_player_cheated()
        # Update the probability table
        if prev_player_cheated:
            self.probs_table["can_play"] = [0, 1, 0, 0]
            self.probs_table["cannot_play"] = [0, 1, 0, 0]
        else:
            self.probs_table = self.base_probs_table.copy()
            self.probs_table.at["call", "can_play"] = 0.0
            self.probs_table.at["call", "cannot_play"] = 0.0
            self.normalize_probs()
        # Call the usual step function
        return super().step(obs, game)


class AdaptiveCheatPlayer(RandomCheatPlayer):
    """Cheat player that adapts the probability of cheating and calling
    based on the size of the pile and the size of opponents' hands.
    """

    def __init__(
        self,
        max_call_prob: float,
        max_cheat_prob: float,
        seed: int = 0,
        is_xray: bool = False,
    ):
        self.max_call_prob = max_call_prob
        self.max_cheat_prob = max_cheat_prob
        self.is_xray = is_xray
        # Start with a naive player, will be adapted each step
        probs_table = pd.DataFrame(
            {"can_play": [0, 0, 0, 1], "cannot_play": [1, 0, 0, 0]},
            index=["pass", "call", "cheat", "play"],
        )
        super().__init__(probs_table=probs_table, seed=seed)

    def step(self, obs: CheatObs, game: CheatGame) -> ActionId:
        """Adapt the probabilities based on the size of the pile and
        opponents' hands, then sample an action.  Specifically, the
        decision of whether to call should be based on a simple estimate
        of the chance that the previous player cheated, which is just
        the chance that the player had a card of an allowed rank in
        their hand.  This chance should be the same whether or not we
        can play.  If we can play and don't call, we should always play.
        If we can't play and don't call, we should choose to cheat based
        on the size of the pile: if it's empty, always cheat, if it
        contains the entire deck, cheat with prob P (provided as
        parameter), and interpolate in between.  Also, some special
        rules: if a card was claimed that you have all instances of in
        your hand, always call.
        """
        # Calculate the call probability
        # Check if the previous action was a played card; if not, we
        # can't call
        deck_size = game.config.num_suits * game.config.num_ranks
        prev_action_s = game.obs_action_meanings.get(
            obs.prev_player_obs_action, None
        )
        if (
            prev_action_s is None
            or prev_action_s == "pass"
            or obs.num_in_pile == 0
        ):
            call_prob = 0.0
        else:
            # We can call, so check if we have x-ray powers and the last
            # play was a cheat
            if self.is_xray:
                if game.last_player_cheated():
                    call_prob = 1.0
                else:
                    call_prob = 0.0
            else:
                # Check if all the instances of the last claimed card
                # are in our hand
                num_in_hand_matching_claim = len(
                    [
                        card
                        for card in obs.cards_in_hand
                        if card.rank == obs.last_claimed_rank
                    ]
                )
                # Note: a max_call_prob of exactly zero will prevent
                # even definite calls
                if (
                    num_in_hand_matching_claim == game.config.num_suits
                    and self.max_call_prob != 0.0
                ):
                    call_prob = 1.0
                else:
                    # We shouldn't definitely call, so calculate an
                    # approximate probability
                    call_prob = self.max_call_prob * (
                        1 - (obs.num_in_other_hands[0] / deck_size)
                    ) ** (game.config.num_suits - num_in_hand_matching_claim)
        # Calculate the cheat probability based on the size of the pile
        cheat_prob = self.max_cheat_prob * (1 - (obs.num_in_pile / deck_size))
        # Update the probability table
        self.probs_table["can_play"] = [0, call_prob, 0, 1 - call_prob]
        self.probs_table["cannot_play"] = [
            (1 - call_prob) * (1 - cheat_prob),
            call_prob,
            (1 - call_prob) * cheat_prob,
            0,
        ]
        # Run a normal step
        return super().step(obs, game)


def run(
    game: CheatGame,
    players: list[CheatPlayer],
    max_turns: int = 1000,
    verbose: bool = False,
    show_progress: bool = False,
    seed: Optional[int] = None,
    equal_turns: bool = True,
):
    """Run a cheat game to termination, with an optional max number
    of turns (turns, not rounds)."""
    assert len(players) == game.config.num_players

    # Reset the game and players with different seeds if specified
    current_player, obs, winning_player = game.reset(seed=seed)
    for player_idx, player in enumerate(players):
        player.reset(seed=seed if seed is None else seed + player_idx)

    # Loop through turns
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

    # Take extra dummy turns if needed to get equal number of turns for
    # all players, if specified
    if equal_turns:
        while (len(game.state_history) % game.config.num_players) != 0:
            _ = game.step("pass")

    scores = game.get_player_scores()
    if verbose:
        print("DONE -----------------------------------")
        print(f"scores: {', '.join([f'{score:0.2f}' for score in scores])}")
        if winning_player is not None:
            print(f"Player {winning_player} won by discarding all cards!")
    return scores, winning_player, turn_cnt

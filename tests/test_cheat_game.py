"""Test suite for Cheat game objects."""

from sortedcontainers import SortedList
from ai_safety_games import cheat, utils
from ai_safety_games.cheat import (
    Card,
    CheatObs,
    CheatState,
)  # For easier testing

utils.enable_ipython_reload()


def test_cheat_game():
    # Create the game
    game = cheat.CheatGame(
        config=cheat.CheatConfig(num_ranks=4, num_suits=2, seed=0)
    )

    # Check some stuff
    assert game.action_meanings == {
        0: "deal",
        1: "pass",
        2: "call",
        3: "c00_p00",
        4: "c00_p01",
        5: "c00_p02",
        6: "c00_p03",
        7: "c01_p00",
        8: "c01_p01",
        9: "c01_p02",
        10: "c01_p03",
        11: "c02_p00",
        12: "c02_p01",
        13: "c02_p02",
        14: "c02_p03",
        15: "c03_p00",
        16: "c03_p01",
        17: "c03_p02",
        18: "c03_p03",
    }
    assert game.action_ids == {
        "deal": 0,
        "pass": 1,
        "call": 2,
        "c00_p00": 3,
        "c00_p01": 4,
        "c00_p02": 5,
        "c00_p03": 6,
        "c01_p00": 7,
        "c01_p01": 8,
        "c01_p02": 9,
        "c01_p03": 10,
        "c02_p00": 11,
        "c02_p01": 12,
        "c02_p02": 13,
        "c02_p03": 14,
        "c03_p00": 15,
        "c03_p01": 16,
        "c03_p02": 17,
        "c03_p03": 18,
    }
    assert game.obs_action_ids == {
        "deal": 0,
        "pass": 1,
        "call": 2,
        "c00": 3,
        "c01": 4,
        "c02": 5,
        "c03": 6,
    }
    assert game.obs_action_meanings == {
        0: "deal",
        1: "pass",
        2: "call",
        3: "c00",
        4: "c01",
        5: "c02",
        6: "c03",
    }
    assert game.cards_to_string([Card(2, 3), Card(0, 0)]) == "2|3, 0|0"

    # Tick through a few turns manually, confirming state as we go
    # Reset
    current_player, obs, winning_player = game.reset()
    assert game.state == CheatState(
        hands=[
            SortedList(
                [
                    Card(rank=0, suit=1),
                    Card(rank=1, suit=1),
                    Card(rank=3, suit=1),
                ]
            ),
            SortedList([Card(rank=0, suit=0), Card(rank=3, suit=0)]),
            SortedList(
                [
                    Card(rank=1, suit=0),
                    Card(rank=2, suit=0),
                    Card(rank=2, suit=1),
                ]
            ),
        ],
        pile=[],
        current_player=2,
        prev_player_action=0,
        prev_player_effective_action=1,
        allowed_next_ranks=SortedList([0, 1, 2, 3]),
        num_continuous_passes=0,
        burned_cards=[],
    )
    assert obs == CheatObs(
        cards_in_hand=SortedList(
            [
                Card(rank=1, suit=0),
                Card(rank=2, suit=0),
                Card(rank=2, suit=1),
            ]
        ),
        num_in_other_hands=[3, 2],
        num_in_pile=0,
        prev_player_obs_action=1,
        allowed_next_ranks=SortedList([0, 1, 2, 3]),
        num_continuous_passes=0,
        num_burned_cards=0,
    )
    # Player 2 plays a card and doesn't cheat
    current_player, obs, winning_player = game.step("c01_p01")
    assert game.state == CheatState(
        hands=[
            SortedList(
                [
                    Card(rank=0, suit=1),
                    Card(rank=1, suit=1),
                    Card(rank=3, suit=1),
                ]
            ),
            SortedList([Card(rank=0, suit=0), Card(rank=3, suit=0)]),
            SortedList([Card(rank=2, suit=0), Card(rank=2, suit=1)]),
        ],
        pile=[Card(rank=1, suit=0)],
        current_player=0,
        prev_player_action=8,
        prev_player_effective_action=8,
        allowed_next_ranks=SortedList([2]),
        num_continuous_passes=0,
        burned_cards=[],
    )
    assert obs == CheatObs(
        cards_in_hand=SortedList(
            [Card(rank=0, suit=1), Card(rank=1, suit=1), Card(rank=3, suit=1)]
        ),
        num_in_other_hands=[2, 2],
        num_in_pile=1,
        prev_player_obs_action=4,
        allowed_next_ranks=SortedList([2]),
        num_continuous_passes=0,
        num_burned_cards=0,
    )
    # Player 0 calls, has to pick up the card
    current_player, obs, winning_player = game.step("call")
    assert game.state == CheatState(
        hands=[
            SortedList(
                [
                    Card(rank=0, suit=1),
                    Card(rank=1, suit=0),
                    Card(rank=1, suit=1),
                    Card(rank=3, suit=1),
                ]
            ),
            SortedList([Card(rank=0, suit=0), Card(rank=3, suit=0)]),
            SortedList([Card(rank=2, suit=0), Card(rank=2, suit=1)]),
        ],
        pile=[],
        current_player=1,
        prev_player_action=2,
        prev_player_effective_action=2,
        allowed_next_ranks=SortedList([0, 1, 2, 3]),
        num_continuous_passes=0,
        burned_cards=[],
    )
    assert obs == CheatObs(
        cards_in_hand=SortedList([Card(rank=0, suit=0), Card(rank=3, suit=0)]),
        num_in_other_hands=[2, 4],
        num_in_pile=0,
        prev_player_obs_action=2,
        allowed_next_ranks=SortedList([0, 1, 2, 3]),
        num_continuous_passes=0,
        num_burned_cards=0,
    )
    # Player 1 plays a 0 and claims a 3, cheating
    current_player, obs, winning_player = game.step("c03_p00")
    assert game.state == CheatState(
        hands=[
            SortedList(
                [
                    Card(rank=0, suit=1),
                    Card(rank=1, suit=0),
                    Card(rank=1, suit=1),
                    Card(rank=3, suit=1),
                ]
            ),
            SortedList([Card(rank=3, suit=0)]),
            SortedList([Card(rank=2, suit=0), Card(rank=2, suit=1)]),
        ],
        pile=[Card(rank=0, suit=0)],
        current_player=2,
        prev_player_action=15,
        prev_player_effective_action=15,
        allowed_next_ranks=SortedList([0]),
        num_continuous_passes=0,
        burned_cards=[],
    )
    assert obs == CheatObs(
        cards_in_hand=SortedList([Card(rank=2, suit=0), Card(rank=2, suit=1)]),
        num_in_other_hands=[4, 1],
        num_in_pile=1,
        prev_player_obs_action=6,
        allowed_next_ranks=SortedList([0]),
        num_continuous_passes=0,
        num_burned_cards=0,
    )
    # Player 2 calls, successfully
    current_player, obs, winning_player = game.step("call")
    assert game.state == CheatState(
        hands=[
            SortedList(
                [
                    Card(rank=0, suit=1),
                    Card(rank=1, suit=0),
                    Card(rank=1, suit=1),
                    Card(rank=3, suit=1),
                ]
            ),
            SortedList([Card(rank=0, suit=0), Card(rank=3, suit=0)]),
            SortedList([Card(rank=2, suit=0), Card(rank=2, suit=1)]),
        ],
        pile=[],
        current_player=2,
        prev_player_action=2,
        prev_player_effective_action=2,
        allowed_next_ranks=SortedList([0, 1, 2, 3]),
        num_continuous_passes=0,
        burned_cards=[],
    )
    assert obs == CheatObs(
        cards_in_hand=SortedList([Card(rank=2, suit=0), Card(rank=2, suit=1)]),
        num_in_other_hands=[4, 2],
        num_in_pile=0,
        prev_player_obs_action=2,
        allowed_next_ranks=SortedList([0, 1, 2, 3]),
        num_continuous_passes=0,
        num_burned_cards=0,
    )
    # Player 2 plays a card
    current_player, obs, winning_player = game.step("c02_p02")
    assert game.state == CheatState(
        hands=[
            SortedList(
                [
                    Card(rank=0, suit=1),
                    Card(rank=1, suit=0),
                    Card(rank=1, suit=1),
                    Card(rank=3, suit=1),
                ]
            ),
            SortedList([Card(rank=0, suit=0), Card(rank=3, suit=0)]),
            SortedList([Card(rank=2, suit=1)]),
        ],
        pile=[Card(rank=2, suit=0)],
        current_player=0,
        prev_player_action=13,
        prev_player_effective_action=13,
        allowed_next_ranks=SortedList([3]),
        num_continuous_passes=0,
        burned_cards=[],
    )
    assert obs == CheatObs(
        cards_in_hand=SortedList(
            [
                Card(rank=0, suit=1),
                Card(rank=1, suit=0),
                Card(rank=1, suit=1),
                Card(rank=3, suit=1),
            ]
        ),
        num_in_other_hands=[2, 1],
        num_in_pile=1,
        prev_player_obs_action=5,
        allowed_next_ranks=SortedList([3]),
        num_continuous_passes=0,
        num_burned_cards=0,
    )
    # Player 0 plays a card
    current_player, obs, winning_player = game.step("c03_p03")
    assert game.state == CheatState(
        hands=[
            SortedList(
                [
                    Card(rank=0, suit=1),
                    Card(rank=1, suit=0),
                    Card(rank=1, suit=1),
                ]
            ),
            SortedList([Card(rank=0, suit=0), Card(rank=3, suit=0)]),
            SortedList([Card(rank=2, suit=1)]),
        ],
        pile=[Card(rank=2, suit=0), Card(rank=3, suit=1)],
        current_player=1,
        prev_player_action=18,
        prev_player_effective_action=18,
        allowed_next_ranks=SortedList([0]),
        num_continuous_passes=0,
        burned_cards=[],
    )
    assert obs == CheatObs(
        cards_in_hand=SortedList([Card(rank=0, suit=0), Card(rank=3, suit=0)]),
        num_in_other_hands=[1, 3],
        num_in_pile=2,
        prev_player_obs_action=6,
        allowed_next_ranks=SortedList([0]),
        num_continuous_passes=0,
        num_burned_cards=0,
    )
    # Player 1 passes
    current_player, obs, winning_player = game.step("pass")
    assert game.state == CheatState(
        hands=[
            SortedList(
                [
                    Card(rank=0, suit=1),
                    Card(rank=1, suit=0),
                    Card(rank=1, suit=1),
                ]
            ),
            SortedList([Card(rank=0, suit=0), Card(rank=3, suit=0)]),
            SortedList([Card(rank=2, suit=1)]),
        ],
        pile=[Card(rank=2, suit=0), Card(rank=3, suit=1)],
        current_player=2,
        prev_player_action=1,
        prev_player_effective_action=1,
        allowed_next_ranks=SortedList([0]),
        num_continuous_passes=1,
        burned_cards=[],
    )
    assert obs == CheatObs(
        cards_in_hand=SortedList([Card(rank=2, suit=1)]),
        num_in_other_hands=[3, 2],
        num_in_pile=2,
        prev_player_obs_action=1,
        allowed_next_ranks=SortedList([0]),
        num_continuous_passes=1,
        num_burned_cards=0,
    )
    # Players 2 and 3 also pass, clearing the pile
    current_player, obs, winning_player = game.step("pass")
    current_player, obs, winning_player = game.step("pass")
    assert game.state == CheatState(
        hands=[
            SortedList(
                [
                    Card(rank=0, suit=1),
                    Card(rank=1, suit=0),
                    Card(rank=1, suit=1),
                ]
            ),
            SortedList([Card(rank=0, suit=0), Card(rank=3, suit=0)]),
            SortedList([Card(rank=2, suit=1)]),
        ],
        pile=[],
        current_player=1,
        prev_player_action=1,
        prev_player_effective_action=1,
        allowed_next_ranks=SortedList([0, 1, 2, 3]),
        num_continuous_passes=0,
        burned_cards=[Card(rank=2, suit=0), Card(rank=3, suit=1)],
    )
    assert obs == CheatObs(
        cards_in_hand=SortedList([Card(rank=0, suit=0), Card(rank=3, suit=0)]),
        num_in_other_hands=[1, 3],
        num_in_pile=0,
        prev_player_obs_action=1,
        allowed_next_ranks=SortedList([0, 1, 2, 3]),
        num_continuous_passes=0,
        num_burned_cards=2,
    )

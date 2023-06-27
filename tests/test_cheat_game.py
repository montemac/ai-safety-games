"""Test suite for Cheat game objects."""
# %%
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
    ASSUMED_ACTION_MEANINGS = {
        0: "pass",
        1: "c00_p00",
        2: "c00_p01",
        3: "c00_p02",
        4: "c00_p03",
        5: "c01_p00",
        6: "c01_p01",
        7: "c01_p02",
        8: "c01_p03",
        9: "c02_p00",
        10: "c02_p01",
        11: "c02_p02",
        12: "c02_p03",
        13: "c03_p00",
        14: "c03_p01",
        15: "c03_p02",
        16: "c03_p03",
        17: "c00_p00_call",
        18: "c00_p01_call",
        19: "c00_p02_call",
        20: "c00_p03_call",
        21: "c01_p00_call",
        22: "c01_p01_call",
        23: "c01_p02_call",
        24: "c01_p03_call",
        25: "c02_p00_call",
        26: "c02_p01_call",
        27: "c02_p02_call",
        28: "c02_p03_call",
        29: "c03_p00_call",
        30: "c03_p01_call",
        31: "c03_p02_call",
        32: "c03_p03_call",
    }
    assert game.action_meanings == ASSUMED_ACTION_MEANINGS
    assert game.action_ids == {
        value: key for key, value in ASSUMED_ACTION_MEANINGS.items()
    }
    ASSUMED_OBS_ACTION_MEANINGS = {
        0: "pass",
        1: "c00",
        2: "c01",
        3: "c02",
        4: "c03",
        5: "c00_call",
        6: "c01_call",
        7: "c02_call",
        8: "c03_call",
    }
    assert game.obs_action_meanings == ASSUMED_OBS_ACTION_MEANINGS
    assert game.obs_action_ids == {
        value: key for key, value in ASSUMED_OBS_ACTION_MEANINGS.items()
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
        prev_player_effective_action=0,
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
        prev_player_obs_action=0,
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
        prev_player_action=6,
        prev_player_effective_action=6,
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
        prev_player_obs_action=2,
        allowed_next_ranks=SortedList([2]),
        num_continuous_passes=0,
        num_burned_cards=0,
    )
    # Player 0 calls, has to pick up the card
    current_player, obs, winning_player = game.step("c00_p00_call")
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
        prev_player_action=17,
        prev_player_effective_action=17,
        allowed_next_ranks=SortedList([0, 1, 2, 3]),
        num_continuous_passes=0,
        burned_cards=[],
    )
    assert obs == CheatObs(
        cards_in_hand=SortedList([Card(rank=0, suit=0), Card(rank=3, suit=0)]),
        num_in_other_hands=[2, 4],
        num_in_pile=0,
        prev_player_obs_action=1,
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
        prev_player_action=13,
        prev_player_effective_action=13,
        allowed_next_ranks=SortedList([0]),
        num_continuous_passes=0,
        burned_cards=[],
    )
    assert obs == CheatObs(
        cards_in_hand=SortedList([Card(rank=2, suit=0), Card(rank=2, suit=1)]),
        num_in_other_hands=[4, 1],
        num_in_pile=1,
        prev_player_obs_action=4,
        allowed_next_ranks=SortedList([0]),
        num_continuous_passes=0,
        num_burned_cards=0,
    )
    # Player 2 calls, successfully, and plays a card
    current_player, obs, winning_player = game.step("c02_p02_call")
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
        prev_player_action=27,
        prev_player_effective_action=27,
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
        prev_player_obs_action=3,
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
        prev_player_action=16,
        prev_player_effective_action=16,
        allowed_next_ranks=SortedList([0]),
        num_continuous_passes=0,
        burned_cards=[],
    )
    assert obs == CheatObs(
        cards_in_hand=SortedList([Card(rank=0, suit=0), Card(rank=3, suit=0)]),
        num_in_other_hands=[1, 3],
        num_in_pile=2,
        prev_player_obs_action=4,
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
        prev_player_action=0,
        prev_player_effective_action=0,
        allowed_next_ranks=SortedList([0]),
        num_continuous_passes=1,
        burned_cards=[],
    )
    assert obs == CheatObs(
        cards_in_hand=SortedList([Card(rank=2, suit=1)]),
        num_in_other_hands=[3, 2],
        num_in_pile=2,
        prev_player_obs_action=0,
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
        prev_player_action=0,
        prev_player_effective_action=0,
        allowed_next_ranks=SortedList([0, 1, 2, 3]),
        num_continuous_passes=0,
        burned_cards=[Card(rank=2, suit=0), Card(rank=3, suit=1)],
    )
    assert obs == CheatObs(
        cards_in_hand=SortedList([Card(rank=0, suit=0), Card(rank=3, suit=0)]),
        num_in_other_hands=[1, 3],
        num_in_pile=0,
        prev_player_obs_action=0,
        allowed_next_ranks=SortedList([0, 1, 2, 3]),
        num_continuous_passes=0,
        num_burned_cards=2,
    )

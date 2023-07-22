"""Test a previously trained cheat model on new games."""
# %%
# Imports, etc.

import pickle
import datetime
import glob
import os

# TEMP
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import pandas as pd
import torch as t
from tqdm.auto import tqdm
import plotly.express as px

from ai_safety_games import cheat, utils, models, training

utils.enable_ipython_reload()

# %%
# Create some random players and a model-based player, run a bunch of
# games

# Load the model-based player
# First model trained on one game at a time (3 sequences, one per
# player)
# MODEL_FN = "cheat_train_output/20230710T080356/model.pkl"

# Second model
DATASET_FOLDER = "../datasets/random_dataset_20230707T111903"
RESULTS_FN = "cheat_results/20230721T134626/results.pkl"

# Load model
with open(RESULTS_FN, "rb") as file:
    results = training.DecisionTransformerTrainingResults(**pickle.load(file))
    model = results.model

# Load data from training dataset for reference, including players and
# summary data from games
with open(os.path.join(DATASET_FOLDER, "config.pkl"), "rb") as file:
    config_dict = pickle.load(file)
    game_config = cheat.CheatConfig(**config_dict["game.config"])
    players_all = [
        cheat.RandomCheatPlayer(player_config["probs_table"])
        for player_config in config_dict["players"]
    ]
with open(os.path.join(DATASET_FOLDER, "summary.pkl"), "rb") as file:
    summary_lists = pickle.load(file)


# Utility function
def scores_to_margins(scores):
    """Convert scores to victory margins."""
    try:
        best_nonwinning_score = max([score for score in scores if score < 0])
    except ValueError:
        return scores
    return [
        score if score < 0 else score - best_nonwinning_score
        for score in scores
    ]


# Print some stats about players
margins_for_games = [
    dict(zip(scores.keys(), scores_to_margins(scores.values())))
    for scores in summary_lists["scores"]
]
all_margins = pd.DataFrame(
    [item for margins in margins_for_games for item in list(margins.items())],
    columns=["player", "margin"],
)
mean_margins_by_player = all_margins.groupby("player").mean()
px.ecdf(mean_margins_by_player, title="CDF of mean margins by player").show()

# %%
# Run a bunch of different games with random opponents, see how the
# model performs
NUM_TEST_GAMES = 2000
RTG_RANGE = [-20, 10]
SEED = 0

# Use this to generate random stuff and sub-seeds for games
rng = np.random.default_rng(seed=SEED)

results_list = []
for game_idx in tqdm(range(NUM_TEST_GAMES)):
    # Randomly choose a goal RTG in the defined range, using a uniform
    # float distribution
    goal_rtg = rng.uniform(*RTG_RANGE)

    # Create a list of players with the model-based player in the first
    # position, then two other randomly-selected players
    players = [
        cheat.ModelCheatPlayer(model=model, goal_rtg=goal_rtg),
        *rng.choice(players_all, size=2, replace=True),
    ]

    # Create the game
    game = cheat.CheatGame(
        config=cheat.CheatConfig(
            num_ranks=13, num_suits=4, seed=rng.integers(1e6)
        )
    )

    # Run the game
    scores, winning_player, turn_cnt = cheat.run(
        game=game,
        players=players,
        max_turns=model.dt_cfg.n_timesteps * game.config.num_players,
        seed=rng.integers(1e6),
    )

    # Calculate victory margins
    margins = scores_to_margins(scores)

    results = {
        "goal_rtg": goal_rtg,
        "game_idx": game_idx,
        "model_margin": margins[0],
    }
    results_list.append(results)

results_df = pd.DataFrame(results_list)

# %%
# Plot and analyze the results

# Get the margins for individual games for the best K players
TOP_K = 50
best_k_players = mean_margins_by_player.sort_values(
    "margin", ascending=False
).index[:TOP_K]
best_k_margins_all = all_margins[all_margins["player"].isin(best_k_players)]
best_k_margins_all = best_k_margins_all.join(
    best_k_margins_all.groupby("player")["margin"].mean(),
    on="player",
    rsuffix="_mean",
)

# Plot scatter of model margin vs. goal RTG, overlay top-K players
fig = px.scatter(
    results_df,
    x="goal_rtg",
    y="model_margin",
    opacity=0.2,
    title="Model victory margin vs goal RTG",
)
fig.add_scatter(
    x=best_k_margins_all["margin_mean"],
    y=best_k_margins_all["margin"],
    mode="markers",
    marker=dict(color="red", opacity=0.01),
    name="Best K players",
)
fig.update_layout(height=600)
fig.show()

# Show histograms of margin for model in certain RTG range, and for
# top-K players that span similar mean margins
rtg_range = (
    best_k_margins_all["margin_mean"].min(),
    best_k_margins_all["margin_mean"].max(),
)
model_good_margins = results_df["model_margin"][
    results_df["goal_rtg"].between(*rtg_range)
]
top_k_good_margins = best_k_margins_all["margin"]
plot_margins = (
    pd.concat(
        [
            model_good_margins.rename("margin").to_frame(),
            top_k_good_margins.to_frame(),
        ],
        keys=["model", "top_k"],
        names=["player"],
    )
    .reset_index()
    .drop("level_1", axis=1)
)
fig = px.histogram(
    plot_margins,
    x="margin",
    color="player",
    histnorm="probability",
    title='Histogram of "good" margins for RTG-conditioned'
    " model and top-K players",
)
fig.update_layout(barmode="overlay")
fig.update_traces(opacity=0.75)
fig.show()


# %%

# PROBS_TABLE = pd.DataFrame(
#     {
#         "can_play": {
#             "pass": 0.0,
#             "call": 0.0,
#             "cheat": 0.0,
#             "play": 1.0,
#         },
#         "cannot_play": {
#             "pass": 1.0,
#             "call": 0.0,
#             "cheat": 0.0,
#             "play": 0.0,
#         },
#     }
# )

# # Best player by number of wins in
# # "../datasets/random_dataset_20230707T111903" dataset
# BEST_PROBS_TABLE_WINS = pd.DataFrame(
#     {
#         "can_play": {
#             "pass": 0.0,
#             "call": 0.0,
#             "cheat": 0.0,
#             "play": 1.0,
#         },
#         "cannot_play": {
#             "pass": 0.526316,
#             "call": 0.052632,
#             "cheat": 0.421053,
#             "play": 0.0,
#         },
#     }
# )

# # Best player by average score in
# # "../datasets/random_dataset_20230707T111903" dataset
# BEST_PROBS_TABLE_SCORE = pd.DataFrame(
#     {
#         "can_play": {
#             "pass": 0.0,
#             "call": 0.0,
#             "cheat": 0.0,
#             "play": 1.0,
#         },
#         "cannot_play": {
#             "pass": 1.0,
#             "call": 0.0,
#             "cheat": 0.0,
#             "play": 0.0,
#         },
#     }
# )

# # Best player by average victory margin in
# # "../datasets/random_dataset_20230707T111903" dataset
# BEST_PROBS_TABLE_MARGIN = pd.DataFrame(
#     {
#         "can_play": {
#             "pass": 0.0,
#             "call": 0.0,
#             "cheat": 0.0,
#             "play": 1.0,
#         },
#         "cannot_play": {
#             "pass": 0.69,
#             "call": 0.034,
#             "cheat": 0.276,
#             "play": 0.0,
#         },
#     }
# )

# List of players
results_list = []
# for goal_rtg in tqdm(range(-20, 2, 2)):
# for goal_rtg in tqdm([-20, -10, 0, 10]):
for goal_rtg in tqdm([-1, 0, 2, 5, 10, 20]):
    # for goal_rtg in [5]:
    players = [
        cheat.ModelCheatPlayer(model=model, goal_rtg=goal_rtg),
        # cheat.RandomCheatPlayer(probs_table=BEST_PROBS_TABLE_MARGIN),
        cheat.RandomCheatPlayer(probs_table=PROBS_TABLE),
        cheat.RandomCheatPlayer(probs_table=PROBS_TABLE),
    ]

    # Create the game
    game = cheat.CheatGame(
        config=cheat.CheatConfig(num_ranks=13, num_suits=4, seed=0)
    )

    # Run the games
    for seed in np.arange(100) + 2000:
        scores, winning_player, turn_cnt = cheat.run(
            game=game,
            players=players,
            max_turns=model.dt_cfg.n_timesteps * game.config.num_players,
            seed=seed,
        )
        results = {
            "goal_rtg": goal_rtg,
            "seed": seed,
        }
        # Calculate victory margins
        best_nonwinning_score = max(
            [
                score
                for player, score in enumerate(scores)
                if player != winning_player
            ]
        )
        victory_margins = [
            score
            if player != winning_player
            else score - best_nonwinning_score
            for player, score in enumerate(scores)
        ]
        # Add scores and victory margins to results
        for player, (score, margin) in enumerate(zip(scores, victory_margins)):
            results[f"score_{player}"] = score
            results[f"margin_{player}"] = margin
        results_list.append(results)

results_df = pd.DataFrame(results_list)

# %%
# Plot results
px.line(
    results_df.groupby("goal_rtg").mean().reset_index(),
    x="goal_rtg",
    y="margin_0",
).show()

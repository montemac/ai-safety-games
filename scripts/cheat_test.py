"""Test a previously trained cheat model on new games."""
# %%
# Imports, etc.

import pickle
import datetime
import glob
import os

# TEMP
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import pandas as pd
import torch as t
from tqdm.auto import tqdm
import plotly.express as px

from ai_safety_games import cheat, utils, training
from ai_safety_games.ScoreTransformer import ScoreTransformer

utils.enable_ipython_reload()

# Disable gradients
_ = t.set_grad_enabled(False)


# %%
# Load some stuff about the dataset, and the model to test

# New embedding scheme
DATASET_FOLDER = "../datasets/random_dataset_20230731T001342"
# 4L attn-only model
# TRAINING_RESULTS_FN = "cheat_results/20230801T095350/results.pkl"
# 1L attn-only model
# TRAINING_RESULTS_FN = "cheat_results/20230801T105951/results.pkl"
# 4L full model
TRAINING_RESULTS_FN = "cheat_results/20230801T111728/results.pkl"

# Load model
with open(TRAINING_RESULTS_FN, "rb") as file:
    results = training.TrainingResults(**pickle.load(file))
    model = results.model

# Create a model description using fields from model.cfg.
# Fields: n_layers, d_model, d_head, n_ctx, attn_only
model_desc = (
    f"{model.cfg.n_layers}L, "
    f"D:{model.cfg.d_model}, "
    f"H:{model.cfg.d_head}, "
    f"C:{model.cfg.n_ctx}, "
    f"A:{model.cfg.attn_only}"
)

# Load data from training dataset for reference, including players and
# summary data from games
with open(os.path.join(DATASET_FOLDER, "config.pkl"), "rb") as file:
    config_dict = pickle.load(file)
    game_config = cheat.CheatConfig(**config_dict["game.config"])
    # Load players (don't save classes in case they have changed a bit
    # and loading breaks)
    players_all = []
    for class_str, player_vars in config_dict["players"]:
        class_name = class_str.split("'")[1].split(".")[-1]
        if class_name == "NaiveCheatPlayer":
            player = cheat.NaiveCheatPlayer()
        elif class_name == "XRayCheatPlayer":
            player = cheat.XRayCheatPlayer(
                probs_table=player_vars["probs_table"]
            )
        elif class_name == "AdaptiveCheatPlayer":
            player = cheat.AdaptiveCheatPlayer(
                max_call_prob=player_vars["max_call_prob"],
                max_cheat_prob=player_vars["max_cheat_prob"],
            )
        players_all.append(player)
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
# px.ecdf(mean_margins_by_player, title="CDF of mean margins by player").show()
# px.histogram(
#     all_margins["margin"], title="Histogram of margins in training data"
# ).show()
# px.ecdf(all_margins["margin"], title="CDF of all margins").show()


# %%
# Run a bunch of different games with random opponents, see how the
# model performs
GAMES_PER_SCORE = 100
GOAL_SCORES = np.arange(-4, 4)
SEED = 0
NO_XRAY_OPP = True

# Use this to generate random stuff and sub-seeds for games
rng = np.random.default_rng(seed=SEED)

results_list = []
game_pbar = tqdm(total=GAMES_PER_SCORE)
for goal_score in tqdm(GOAL_SCORES):
    for game_idx in range(GAMES_PER_SCORE):
        # Create the game
        game = cheat.CheatGame(config=game_config)
        vocab, _ = game.get_token_vocab()

        # Players to choose from
        if NO_XRAY_OPP:
            players_can_use = [
                player
                for player in players_all
                if not isinstance(player, cheat.XRayCheatPlayer)
            ]
        else:
            players_can_use = players_all

        # Create a list of players with the model-based player in the first
        # position, then two other randomly-selected players
        players = [
            cheat.ScoreTransformerCheatPlayer(
                model=model, vocab=vocab, goal_score=goal_score
            ),
            *rng.choice(players_can_use, size=2, replace=True),
        ]

        # Run the game
        scores, winning_player, turn_cnt = cheat.run(
            game=game,
            players=players,
            max_turns=max(summary_lists["turn_cnt"]) + 1,
            seed=rng.integers(1e6),
        )

        # Calculate victory margins
        margins = scores_to_margins(scores)

        results = {
            "goal_score": goal_score,
            "game_idx": game_idx,
            "model_margin": margins[0],
        }
        results_list.append(results)

        game_pbar.update(1)
    game_pbar.reset()
game_pbar.close()

results_df = pd.DataFrame(results_list)

# %%
# Plot and analyze the results

# # Get the margins for individual games for the best K players
# TOP_K = 10
# best_k_players = mean_margins_by_player.sort_values(
#     "margin", ascending=False
# ).index[:TOP_K]
# best_k_margins_all = all_margins[all_margins["player"].isin(best_k_players)]
# best_k_margins_all = best_k_margins_all.join(
#     best_k_margins_all.groupby("player")["margin"].mean(),
#     on="player",
#     rsuffix="_mean",
# )


# Plot scatter of model margin vs. goal RTG, overlay top-K players
fig = px.scatter(
    results_df,
    x="goal_score",
    y="model_margin",
    opacity=0.2,
    title=f"Model victory margin vs goal score<br>{model_desc}",
)
# fig.add_scatter(
#     x=best_k_margins_all["margin_mean"],
#     y=best_k_margins_all["margin"],
#     mode="markers",
#     marker=dict(color="red", opacity=0.01),
#     name="Best K players",
# )
mean_margin = results_df.groupby("goal_score")["model_margin"].mean()
fig.add_scatter(x=mean_margin.index, y=mean_margin, name="Mean actual score")
fig.add_scatter(
    x=GOAL_SCORES, y=GOAL_SCORES, name="Perfect score conditioning"
)
fig.update_layout(
    height=600, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)
fig.show()


# # Show histograms of margin for model in certain RTG range, and for
# # top-K players that span similar mean margins
# rtg_range = (
#     best_k_margins_all["margin_mean"].min(),
#     best_k_margins_all["margin_mean"].max(),
# )
# model_good_margins = results_df["model_margin"][
#     results_df["goal_rtg"].between(*rtg_range)
# ]
# top_k_good_margins = best_k_margins_all["margin"]
# plot_margins = (
#     pd.concat(
#         [
#             model_good_margins.rename("margin").to_frame(),
#             top_k_good_margins.to_frame(),
#         ],
#         keys=["model", "top_k"],
#         names=["player"],
#     )
#     .reset_index()
#     .drop("level_1", axis=1)
# )
# fig = px.histogram(
#     plot_margins,
#     x="margin",
#     color="player",
#     histnorm="probability",
#     title='Histogram of "good" margins for RTG-conditioned'
#     " model and top-K players",
# )
# fig.update_layout(barmode="overlay")
# fig.update_traces(opacity=0.75)
# fig.show()


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

"""Evaluate and compare different cheat-reduction interventions."""
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
import plotly.graph_objects as go

from ai_safety_games import cheat, utils, training, cheat_utils
from ai_safety_games.ScoreTransformer import ScoreTransformer

utils.enable_ipython_reload()

# Disable gradients
_ = t.set_grad_enabled(False)

game_filter = None

GOAL_SCORE = 5
# TODO: this should be in config somewhere
MAX_TURNS = 40
NUM_GAMES = 2000


# %%
# Filtering of training data
# ----------------------------------------------------------------------------
FOLDERS = (
    [
        "20230828T152710",
        "20230828T153956",
        "20230828T155240",
        "20230828T160534",
        "20230828T161917",
        "20230828T163421",
    ]
    + [
        "20230828T202053",
        "20230828T203332",
        "20230828T204618",
        "20230828T205926",
        "20230828T211249",
    ]
    + [
        "20230828T230224",
        "20230828T231500",
        "20230828T232745",
        "20230828T234022",
    ]
)
INCLUDE_CHEAT_RATES = (
    [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
    + [
        0.02,
        0.05,
        0.15,
        0.2,
        0.25,
    ]
    + [0.001, 0.002, 0.004, 0.007]
)

filter_test_results_list = []
for folder, include_cheat_rate in tqdm(
    list(zip(FOLDERS, INCLUDE_CHEAT_RATES))
):
    # Load training results
    with open(
        os.path.join("cheat_train_results", folder, "results.pkl"), "rb"
    ) as file:
        results = pickle.load(file)

    # Load info about the dataset
    (
        game_config,
        players_all,
    ) = cheat_utils.load_config_and_players_from_dataset(
        results["config"]["dataset_folder"]
    )

    # Create test players list
    test_players = [
        players_all[idx] for idx in results["config"]["test_player_inds"]
    ]

    # Play a bunch of test games and store results
    margins, cheat_rate = cheat_utils.run_test_games(
        model=results["training_results"]["model"],
        game_config=game_config,
        num_games=NUM_GAMES,
        goal_score=GOAL_SCORE,
        max_turns=MAX_TURNS,
        players=test_players,
        seed=0,
        show_progress=True,
        map_func=cheat_utils.get_action_types,
        reduce_func=cheat_utils.get_cheat_rate,
    )
    filter_test_results_list.append(
        {
            "include_cheat_rate": include_cheat_rate,
            "goal_score": GOAL_SCORE,
            "win_rate": (margins > 0).mean(),
            "cheat_rate": cheat_rate,
        }
    )

filter_test_results_df = pd.DataFrame(filter_test_results_list).sort_values(
    "include_cheat_rate"
)

perf_data_filter = filter_test_results_df[
    ["cheat_rate", "win_rate", "include_cheat_rate"]
].reset_index(drop=True)


# %%
# Penalties on cheat action log-probs during training
# ----------------------------------------------------------------------------

# With bug that INCREASED the cheat rates!
# RESULTS_RANGE = ("20230827T002903", "20230827T060501")
RESULTS_RANGE = ("20230827T234948", "20230828T053121")

# Get folders for results in range
fns_all = list(glob.glob("cheat_train_results/*"))
fns_in_range = [
    fn
    for fn in fns_all
    if RESULTS_RANGE[0] <= fn.split("/")[-1] <= RESULTS_RANGE[1]
]

# Load one-by-one and store key results
results_list = []
penalize_test_results_list = []
for fn in tqdm(fns_in_range):
    # Load training results
    with open(os.path.join(fn, "results.pkl"), "rb") as file:
        results = pickle.load(file)
    results_list.append(
        pd.DataFrame(
            {
                "fn": fn,
                "cheat_penalty_weight": results["config"][
                    "cheat_penalty_weight"
                ],
                "cheat_penalty_apply_prob": results["config"][
                    "cheat_penalty_apply_prob"
                ],
                "win_rate": results["training_results"]["results"][
                    "test_win_rate_goal_5"
                ],
            }
        )
    )

    # Load info about the dataset
    (
        game_config,
        players_all,
    ) = cheat_utils.load_config_and_players_from_dataset(
        results["config"]["dataset_folder"]
    )

    # Create test players list
    test_players = [
        players_all[idx] for idx in results["config"]["test_player_inds"]
    ]

    # Play a bunch of test games and store results
    margins, cheat_rate = cheat_utils.run_test_games(
        model=results["training_results"]["model"],
        game_config=game_config,
        num_games=NUM_GAMES,
        goal_score=GOAL_SCORE,
        max_turns=MAX_TURNS,
        players=test_players,
        seed=0,
        show_progress=True,
        map_func=cheat_utils.get_action_types,
        reduce_func=cheat_utils.get_cheat_rate,
    )
    penalize_test_results_list.append(
        {
            "cheat_penalty_weight": results["config"]["cheat_penalty_weight"],
            "cheat_penalty_apply_prob": results["config"][
                "cheat_penalty_apply_prob"
            ],
            "goal_score": GOAL_SCORE,
            "win_rate": (margins > 0).mean(),
            "cheat_rate": cheat_rate,
        }
    )

penalize_test_results_df = pd.DataFrame(
    penalize_test_results_list
).sort_values(["cheat_penalty_weight", "cheat_penalty_apply_prob"])

# results_df = (
#     pd.concat(results_list)
#     .reset_index()
#     .sort_values(["cheat_penalty_weight", "cheat_penalty_apply_prob"])
# )
# px.line(
#     results_df,
#     x="index",
#     y="win_rate",
#     color="cheat_penalty_weight",
#     facet_col="cheat_penalty_apply_prob",
# )

# %%
# Visualize results
plot_df = penalize_test_results_df.melt(
    id_vars=["cheat_penalty_weight", "cheat_penalty_apply_prob"],
    value_vars=["win_rate", "cheat_rate"],
).rename({"variable": "quantity", "value": "rate"}, axis=1)

# px.line(
#     plot_df,
#     x="cheat_penalty_weight",
#     y="rate",
#     color="cheat_penalty_apply_prob",
#     facet_col="quantity",
#     log_x=True,
# ).show()

# px.scatter(
#     penalize_test_results_df,
#     x="cheat_rate",
#     y="win_rate",
#     color=np.log10(test_results_df["cheat_penalty_weight"]),
#     size=np.log10(test_results_df["cheat_penalty_apply_prob"]) + 3,
#     hover_data=["cheat_penalty_weight", "cheat_penalty_apply_prob"],
# ).show()

px.line(
    penalize_test_results_df,
    x="cheat_rate",
    y="win_rate",
    # color=np.log10(test_results_df["cheat_penalty_weight"]),
    color="cheat_penalty_apply_prob",
    title="Penalize cheat action log-probs during training",
).show()

perf_data_penalize = penalize_test_results_df[
    ["cheat_rate", "win_rate"]
].reset_index()

# %%
# Vary score conditioning
# ----------------------------------------------------------------------------
TRAINING_RESULTS_FN = "cheat_train_results/20230817T151856/results.pkl"

with open(TRAINING_RESULTS_FN, "rb") as file:
    results_all = pickle.load(file)
results = training.TrainingResults(**results_all["training_results"])
model = results.model
# Fix
if "n_ctx" not in results_all["config"]:
    results_all["config"]["n_ctx"] = model.cfg.n_ctx
config = cheat_utils.CheatTrainingConfig(**results_all["config"])

game_config, players_all = cheat_utils.load_config_and_players_from_dataset(
    config.dataset_folder
)

test_game_config = cheat.CheatConfig(**vars(game_config))
test_game_config.penalize_wrong_played_card = True

margins_list = []
cheat_rates_list = []
# for player in [results.model] + test_players:
player = model
for goal_score in tqdm(np.linspace(-GOAL_SCORE, GOAL_SCORE, 20)):
    margins, cheat_rate = cheat_utils.run_test_games(
        model=player,
        game_config=game_config,
        num_games=NUM_GAMES,
        goal_score=goal_score,
        max_turns=MAX_TURNS,
        players=test_players,
        seed=0,
        show_progress=True,
        map_func=cheat_utils.get_action_types,
        reduce_func=cheat_utils.get_cheat_rate,
    )
    margins_list.append(
        {
            "goal_score": goal_score,
            "mean_margin": margins.mean(),
            "win_rate": (margins > 0).mean(),
        }
    )
    cheat_rates_list.append(
        {"goal_score": goal_score, "cheat_rate": cheat_rate}
    )
margins_df = pd.DataFrame(margins_list)
cheat_rates = pd.DataFrame(cheat_rates_list).set_index("goal_score")[
    "cheat_rate"
]

perf_data_score = pd.DataFrame(
    {
        "cheat_rate": cheat_rates,
        "win_rate": margins_df.set_index("goal_score")["win_rate"],
    }
).reset_index()


# %%
# Pass logit offsets
# ----------------------------------------------------------------------------
game = cheat.CheatGame(config=game_config)
vocab, action_vocab = game.get_token_vocab()
vocab_str = {idx: token_str for token_str, idx in vocab.items()}

# Token properties table
token_props, action_props = cheat.get_token_props(vocab=vocab)


# Add a pytorch forward hook to the model to add an offset to the
# "pass" action logits
def make_pass_offset_hook(offset):
    def pass_offset_hook(module, input, output):
        output[:, :, vocab["a_pass"]] += offset

    return pass_offset_hook


win_rates = []
cheat_rates_list = []
for pass_offset in tqdm(np.linspace(-1, 5, 20)):
    # Register hook
    handle = model.register_forward_hook(make_pass_offset_hook(pass_offset))
    try:
        # Run games
        margins, cheat_rate = cheat_utils.run_test_games(
            model=model,
            game_config=game_config,
            num_games=NUM_GAMES,
            goal_score=GOAL_SCORE,
            max_turns=max(
                results_all["config"]["cached_game_data"]["summary"][
                    "turn_cnt"
                ]
            ),
            players=test_players,
            seed=0,
            show_progress=True,
            map_func=cheat_utils.get_action_types,
            reduce_func=cheat_utils.get_cheat_rate,
        )
    finally:
        # Clear all forward hooks
        handle.remove()

    win_rate = (margins > 0).mean()
    win_rates.append({"pass_offset": pass_offset, "win_rate": win_rate})
    cheat_rates_list.append(
        {"pass_offset": pass_offset, "cheat_rate": cheat_rate}
    )
win_rates_df = pd.DataFrame(win_rates)
cheat_rates_df = pd.DataFrame(cheat_rates_list)

perf_data_offset = pd.DataFrame(
    {
        "cheat_rate": cheat_rates_df.set_index("pass_offset")["cheat_rate"],
        "win_rate": win_rates_df.set_index("pass_offset")["win_rate"],
    }
).reset_index()


# %%
# Save all results for later
# ----------------------------------------------------------------------------
perf_data = pd.concat(
    [
        perf_data_filter.assign(intervention="filter training data"),
        perf_data_penalize.assign(intervention="penalize in training"),
        perf_data_score.assign(intervention="change prompt"),
        perf_data_offset.assign(intervention="increase pass prob")[
            perf_data_offset["pass_offset"] > -0.1
        ],
    ]
)

save_dict = {"perf_data": perf_data, "penalize_data": penalize_test_results_df}

with open("cheat_eval_results.pkl", "wb") as file:
    pickle.dump(save_dict, file)


# %%
# Now plot all the results
# ----------------------------------------------------------------------------

# Load results
with open("cheat_eval_results.pkl", "rb") as file:
    save_dict = pickle.load(file)
perf_data = save_dict["perf_data"]
penalize_test_results_df = save_dict["penalize_data"]

# Pareto front
perf_sorted = perf_data.sort_values(["cheat_rate"])
perf_pareto = perf_sorted[
    perf_sorted["win_rate"] >= perf_sorted["win_rate"].cummax()
]


# Plot all intervention results
fig = px.line(
    perf_data,
    x="cheat_rate",
    y="win_rate",
    color="intervention",
    title="Effects of deception-reducing interventions",
    labels={"cheat_rate": "Deception rate", "win_rate": "Win rate"},
    log_x=True,
    log_y=True,
)
fig.update_traces(mode="markers")
# Add a dashed line tracing the pareto front
fig.add_trace(
    go.Scatter(
        x=perf_pareto["cheat_rate"],
        y=perf_pareto["win_rate"],
        mode="lines",
        line=dict(color="black", dash="dash"),
        showlegend=True,
        name="Pareto front",
    )
)
# Make x and y axes show percentages with 1 decimal place
fig.update_layout(
    # xaxis_tickformat=".1%",
    # yaxis_tickformat=".0%",
    width=700,
    # legend=dict(
    #     orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    # )
)
fig.show()

# Plots interesting stuff for each intervention
# Training data filtering
fig = px.scatter(
    perf_data[
        (perf_data["intervention"] == "filter training data")
        & (perf_data["include_cheat_rate"] > 0)
    ],
    x="include_cheat_rate",
    y="cheat_rate",
    log_x=True,
    trendline="ols",
    trendline_options=dict(log_x=True),
    title="Cheat rate vs chance of including a game with cheating in training",
    labels={
        "cheat_rate": "Deception rate",
        "include_cheat_rate": "Cheat game inclusion probability",
    },
)
fig.update_layout(
    xaxis_tickformat=".1%",
    yaxis_tickformat=".1%",
    width=700,
)
fig.show()
# Penalize in training
for param, param_name in [
    ("cheat_penalty_weight", "cheat penalty weight"),
    ("cheat_penalty_apply_prob", "cheat penalty apply prob"),
]:
    fig = px.scatter(
        penalize_test_results_df,
        x="cheat_rate",
        y="win_rate",
        color=np.log10(penalize_test_results_df[param]),
        title=f"Effect of different {param_name}s during training",
        labels={"cheat_rate": "Deception rate", "win_rate": "Win rate"},
    )
    # Add a dashed line tracing the pareto front
    fig.add_trace(
        go.Scatter(
            x=perf_pareto["cheat_rate"],
            y=perf_pareto["win_rate"],
            mode="lines",
            line=dict(color="black", dash="dash"),
            showlegend=False,
            name="Pareto front",
        )
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=param_name,
            tickvals=np.log10(penalize_test_results_df[param].unique()),
            ticktext=[
                f"{val:0.2f}"
                for val in penalize_test_results_df[param].unique()
            ],
        )
    )
    fig.update_layout(
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        width=700,
    )
    fig.show()
# Change goal score
fig = px.line(
    perf_data[perf_data["intervention"] == "change goal score"],
    x="goal_score",
    y=["cheat_rate", "win_rate"],
    labels={"goal_score": "Goal score", "value": "Rate"},
    title="Effect of changing goal score",
)
fig.update_layout(
    yaxis_tickformat=".1%",
    width=700,
)
fig.show()

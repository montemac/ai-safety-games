"""Interpret a cheat model."""
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
from einops import rearrange, einsum
import xarray as xr

from ai_safety_games import cheat, utils, training, cheat_utils
from ai_safety_games.ScoreTransformer import ScoreTransformer

utils.enable_ipython_reload()

# Disable gradients
_ = t.set_grad_enabled(False)


# %%
# Load some stuff about the dataset, and the model to test

# 4L attn-only model
# TRAINING_RESULTS_FN = "cheat_results/20230801T095350/results.pkl"
# 1L attn-only model
# TRAINING_RESULTS_FN = "cheat_results/20230801T105951/results.pkl"
# 4L full model
# TRAINING_RESULTS_FN = "cheat_results/20230801T111728/results.pkl"
# 4L full model, separate score and BOG embeddings
# TRAINING_RESULTS_FN = "cheat_results/20230801T123246/results.pkl"
# 8L full model, separate score and BOG embeddings
# TRAINING_RESULTS_FN = "cheat_results/20230801T130838/results.pkl"
#
# TRAINING_RESULTS_FN = "cheat_train_results/20230815T153630/results.pkl"
# Trained on win only
# TRAINING_RESULTS_FN = "cheat_train_results/20230815T234841/results.pkl"
# Small model, trained only on naive and 0.25/1.0 adaptive
TRAINING_RESULTS_FN = "cheat_train_results/20230817T151856/results.pkl"

# Load model

# TODO: fix this problem with loading models!
# AttributeError: Can't get attribute 'game_filter' on
game_filter = None

with open(TRAINING_RESULTS_FN, "rb") as file:
    results_all = pickle.load(file)
config = cheat_utils.CheatTrainingConfig(**results_all["config"])
results = training.TrainingResults(**results_all["training_results"])
model = results.model

# Do some interp-friendly processing of weights
model.process_weights_(
    fold_ln=True, center_writing_weights=False, center_unembed=True
)

with open(os.path.join(config.dataset_folder, "config.pkl"), "rb") as file:
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

game = cheat.CheatGame(config=game_config)
vocab, action_vocab = game.get_token_vocab()
vocab_str = {idx: token_str for token_str, idx in vocab.items()}

# Positions corresponding to actions
first_action_pos = (
    (game_config.num_players - 1) * 2 + 2 + game.config.num_ranks
)
action_pos_step = game_config.num_players * 2 + game.config.num_ranks


# %%
# Create a tokens table containing different token properties
token_props = pd.DataFrame({"id": vocab.values()}, index=vocab.keys())
token_str = pd.Series(vocab.keys(), index=vocab.keys())
# Token type
token_type = pd.Series("misc", index=vocab.keys())
token_type[token_str.str.startswith("a_")] = "player_action"
token_type[token_str.str.match("ar_[\D]")] = "player_result"
token_type[token_str.str.startswith("oa_")] = "other_action"
token_type[token_str.str.match("ar_[\d]")] = "other_result"
token_type[token_str.str.startswith("hand_")] = "hand"
token_props["type"] = token_type
# Data associated with specific token types
# Hand
hand_data = (
    token_str[token_props.type == "hand"]
    .str.extract(r"hand_(\d+)x(\d+)")
    .astype(int)
)
token_props["hand_num"] = -1
token_props.loc[token_props.type == "hand", "hand_num"] = hand_data[0]
token_props["hand_rank"] = -1
token_props.loc[token_props.type == "hand", "hand_rank"] = hand_data[1]
# Claimed card
token_props["claimed_card"] = (
    token_str.str.extract(r".*_c(\d+).*").fillna(-1).astype(int)
)
token_props["played_card"] = (
    token_str.str.extract(r".*_p(\d+).*").fillna(-1).astype(int)
)
# Player the token in associated with
token_player = pd.Series(-1, index=vocab.keys())
is_other_player = token_props.type.isin(["other_action", "other_result"])
token_player[is_other_player] = (
    token_str[is_other_player].str.extract(r"[a-z]+_(\d+)_")[0].astype(int)
)
token_player[
    token_props.type.isin(["player_action", "player_result", "hand"])
] = (game_config.num_players - 1)
token_props["player"] = token_player.astype(int)
# Whether an action is a call or not
token_props["is_call"] = token_str.str.endswith("_call")
# Whether an action is passing or not, or playing an card or not
token_props["is_pass"] = token_str.str.endswith("_pass")
token_props["is_play_card"] = token_str.str.contains("_c[\d]+_")
token_props["is_player_play_card"] = token_props.is_play_card & (
    token_props.type == "player_action"
)
# Whether a player action is cheating or not
is_cheat = pd.Series(False, index=vocab.keys())
is_cheat[token_props.is_player_play_card] = (
    token_props[token_props.is_player_play_card]
    .index.str.extract(".*_c([\d]+)_p([\d]+).*")
    .astype(int)
    .diff(axis=1)[1]
    != 0
)
token_props["is_cheat"] = is_cheat

action_props = token_props[token_props.type == "player_action"]


# %%
# Inspect some of the model weights
W_QK = einsum(
    model.blocks[0].attn.W_Q,
    model.blocks[0].attn.W_K,
    "h dmq dh, h dmk dh -> h dmq dmk",
)
QK = einsum(
    model.token_embed.W_E,
    W_QK,
    model.token_embed.W_E,
    "tq dmq, h dmq dmk, tk dmk -> h tq tk",
)
W_OV = einsum(
    model.blocks[0].attn.W_O,
    model.blocks[0].attn.W_V,
    "h dh dmo, h dmv dh -> h dmo dmv",
)
OV = einsum(
    model.unembed.W_U,
    W_OV,
    model.token_embed.W_E,
    "dmo da, h dmo dmv, tv dmv -> h da tv",
)
# Only look at possible destination tokens that could preceed our
# action, i.e. number of final rank cards
dst_inds = t.tensor(
    [
        vocab[f"hand_{num}x{game_config.num_ranks-1}"]
        for num in range(game_config.num_suits + 1)
    ]
)

# QK matrix to dataframe
QK_df = pd.DataFrame(
    rearrange(QK[:, dst_inds, :].cpu().numpy(), "h tq tk -> tk (h tq)"),
    index=pd.Index(token_props.index, name="src_token"),
    columns=pd.MultiIndex.from_product(
        [range(QK.shape[0]), token_str.iloc[dst_inds]],
        names=["head", "dst_token"],
    ),
)
# Convert the OVf array into a DataFrame where source tokens are rows
# (index) and the columns have a multi-index of the head and the
# action token
OV_df = pd.DataFrame(
    rearrange(OV.cpu().numpy(), "h da tv -> tv (h da)"),
    index=pd.Index(token_props.index, name="src_token"),
    columns=pd.MultiIndex.from_product(
        [range(OV.shape[0]), action_vocab.values()], names=["head", "action"]
    ),
)

# %%
# Visualize some stuff!
filts = {
    "Prev player action": (token_props.type == "other_action")
    & (token_props.player == game_config.num_players - 2),
    "Hand": (token_props.type == "hand"),
}
for filt_name, filt in filts.items():
    fig = px.scatter(
        QK_df[filt].reset_index().melt(id_vars="src_token"),
        x="src_token",
        y="value",
        facet_row="head",
        color="dst_token",
        title=f"QK: {filt_name}",
    )
    fig.show()


# %%
# Run a game, then turn the history into a token sequence and run it through the model
SEED = 0
GOAL_SCORE = 5

rng = np.random.default_rng(seed=SEED)

players_can_use = [players_all[idx] for idx in [0, 3]]

# Create a list of players with the model-based player in the first
# position, then two other randomly-selected players
player_inds_this = rng.choice(len(players_can_use), size=2, replace=True)
players = [
    cheat.ScoreTransformerCheatPlayer(
        model=model,
        vocab=vocab,
        goal_score=GOAL_SCORE,
        temperature=0,
    ),
    *[players_can_use[idx] for idx in player_inds_this],
]

# Run the game
seed = rng.integers(1e6)
scores, winning_player, turn_cnt = cheat.run(
    game=game,
    players=players,
    max_turns=40,
    seed=seed,
)
print(game.print_state_history())

# Turn history into a token sequence
tokens = cheat.get_seqs_from_state_history(
    game=game,
    vocab=vocab,
    state_history=game.state_history,
    players_to_return=[0],
)

# Get logits
model.set_use_attn_result(True)
action_logits, activs = model.run_with_cache(
    tokens=tokens,
    scores=t.tensor([GOAL_SCORE]).to(model.cfg.device),
    remove_batch_dim=True,
)
action_logits = action_logits.squeeze()
action_probs = t.softmax(action_logits, dim=1)
action_sort_inds = action_logits.argsort(dim=1)

tokens = tokens.squeeze()
token_strs = np.array([vocab_str[idx] for idx in tokens.cpu().numpy()])


def plot_attn_and_ov(tokens, token_strs, activs, ov, action_pos, actions):
    """Plot attention pattern for each head, and pattern*OV for each
    head for the provided actions."""
    token_strs = [
        f"{pos}: {token_str}" for pos, token_str in enumerate(token_strs)
    ]
    # Get the attention pattern
    attn_pattern = activs["blocks.0.attn.hook_pattern"][
        :, action_pos - 1, :action_pos
    ].cpu()
    # Get the pattern*OV
    pattern_ov = rearrange(
        attn_pattern[:, None, :]
        * ov[:, actions[:, None], tokens[:action_pos][None, :]].cpu(),
        "h act src -> src act h",
    )
    # Plot the attention pattern
    height = action_pos * 30
    width = 700
    fig = px.imshow(
        pd.DataFrame(
            attn_pattern.T,
            index=pd.Series(token_strs[:action_pos], name="token_str"),
            columns=pd.Series(range(attn_pattern.shape[0]), name="head"),
        ),
        title="Attention pattern",
        color_continuous_scale="piyg",
        color_continuous_midpoint=0,
    )
    fig.update_layout(height=height, width=width / 2)
    fig.show()
    # Plot the pattern*OV
    pattern_ov_da = xr.DataArray(
        pattern_ov,
        dims=["token_str", "action", "head"],
        coords={
            "token_str": token_strs[:action_pos],
            "action": [vocab_str[act.item()] for act in actions],
            "head": range(pattern_ov.shape[-1]),
        },
    )
    fig = px.imshow(
        pattern_ov_da,
        x=pattern_ov_da.coords["action"].values,
        y=pattern_ov_da.coords["token_str"].values,
        facet_col="head",
        title="Pattern*OV",
        color_continuous_scale="piyg",
        color_continuous_midpoint=0,
        labels={"facet_col": "head"},
    )
    fig.update_layout(height=height, width=width)
    fig.show()


action_ind = 1
plot_attn_and_ov(
    tokens=tokens,
    token_strs=token_strs,
    activs=activs,
    ov=OV,
    action_pos=first_action_pos + action_ind * action_pos_step,
    actions=action_sort_inds[action_ind, -5:].flip(dims=[0]),
)

# Visualize the first action choice
# tokens = tokens.squeeze()
# token_strs = np.array([vocab_str[idx] for idx in tokens.cpu().numpy()])
# px.scatter(
#     pd.DataFrame(
#         # QK[:, tokens[first_action_pos - 1], tokens[:first_action_pos]].cpu().T,
#         activs["blocks.0.attn.hook_pattern"][
#             :, first_action_pos - 1, :first_action_pos
#         ]
#         .cpu()
#         .T,
#         index=pd.Series(token_strs[:first_action_pos], name="token_str"),
#     ),
#     facet_col="variable",
#     facet_col_wrap=2,
#     title="QK/pattern for first action",
# ).show()
# px.scatter(
#     pd.DataFrame(
#         OV[:, tokens[first_action_pos], tokens[:first_action_pos]].cpu().T,
#         index=pd.Series(token_strs[:first_action_pos], name="token_str"),
#     ),
#     facet_col="variable",
#     facet_col_wrap=2,
#     title="OV for first action",
# ).show()
# action_to_show = tokens[first_action_pos].item()
# # action_to_show = vocab["a_pass"]
# px.scatter(
#     pd.DataFrame(
#         activs["blocks.0.attn.hook_pattern"][
#             :, first_action_pos - 1, :first_action_pos
#         ]
#         .cpu()
#         .T
#         * OV[:, action_to_show, tokens[:first_action_pos]].cpu().T,
#         index=pd.Series(token_strs[:first_action_pos], name="token_str"),
#     ),
#     facet_col="variable",
#     facet_col_wrap=2,
#     title=f"pattern*OV for first action, action: {vocab_str[action_to_show]}",
# ).show()


# # Get the tokens corresponding to the top K actions
# top_k = 10
# top_k_tokens = pd.DataFrame(
#     [
#         [
#             action_probs[i, idx].item()
#             # vocab_str[idx]
#             for idx in sort_inds[i, -top_k:].cpu().numpy()[::-1]
#         ]
#         for i in range(sort_inds.shape[0])
#     ]
# ).T
# top_k_tokens


# %%
# Get some performance data
test_players = [
    players_all[idx] for idx in results_all["config"]["test_player_inds"]
]

test_game_config = cheat.CheatConfig(**vars(game_config))
test_game_config.penalize_wrong_played_card = True

goal_score = 5
margins_list = []
for player in [results.model] + test_players:
    # for goal_score in [5]:
    margins = cheat_utils.run_test_games(
        model=player,
        game_config=game_config,
        num_games=500,
        goal_score=goal_score,
        max_turns=max(
            results_all["config"]["cached_game_data"]["summary"]["turn_cnt"]
        ),
        players=test_players,
        seed=0,
        show_progress=True,
    )
    margins_list.append(
        {
            "goal_score": goal_score,
            "mean_margin": margins.mean(),
            "win_rate": (margins > 0).mean(),
        }
    )
margins_df = pd.DataFrame(margins_list)
margins_df

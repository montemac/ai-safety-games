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
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
import plotly.express as px
from einops import rearrange, einsum, repeat
import xarray as xr

from ai_safety_games import cheat, utils, training, cheat_utils
from ai_safety_games.ScoreTransformer import ScoreTransformer, IdentityVar

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
# TRAINING_RESULTS_FN = "cheat_train_results/20230817T151856/results.pkl"
# Very small model, 1 head!
# TRAINING_RESULTS_FN = "cheat_train_results/20230831T023701/results.pkl"
# 1 head, 32x32, with EOH
TRAINING_RESULTS_FN = "cheat_train_results/20230907T160608/results.pkl"


# Load model

# TODO: fix this problem with loading models!
# AttributeError: Can't get attribute 'game_filter' on
game_filter = None

with open(TRAINING_RESULTS_FN, "rb") as file:
    results_all = pickle.load(file)
if "n_ctx" in results_all["config"]:
    del results_all["config"]["n_ctx"]
results_all["config"]["max_turns"] = 40  # Wasn't stored in older models
config = cheat_utils.CheatTrainingConfig(**results_all["config"])
results = training.TrainingResults(**results_all["training_results"])
model = results.model

# Do some interp-friendly processing of weights
model.process_weights_(
    fold_ln=True, center_writing_weights=False, center_unembed=True
)

game_config, players_all = cheat_utils.load_config_and_players_from_dataset(
    config.dataset_folder
)

game = cheat.CheatGame(config=game_config)
vocab, action_vocab = game.get_token_vocab(
    include_hand_end=config.include_hand_end
)
vocab_str = {idx: token_str for token_str, idx in vocab.items()}

# Token properties table
token_props, action_props = cheat.get_token_props(vocab=vocab)
token_str = pd.Series(token_props.index, index=token_props.index)


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
# filts = {
#     "Prev player action": (token_props.type == "other_action")
#     & (token_props.player == game_config.num_players - 2),
#     "Hand": (token_props.type == "hand"),
# }
# for filt_name, filt in filts.items():
#     fig = px.scatter(
#         QK_df[filt].reset_index().melt(id_vars="src_token"),
#         x="src_token",
#         y="value",
#         facet_row="head",
#         color="dst_token",
#         title=f"QK: {filt_name}",
#     )
#     fig.show()

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
tokens, first_action_pos, action_pos_step = cheat.get_seqs_from_state_history(
    game=game,
    vocab=vocab,
    state_history=game.state_history,
    players_to_return=[0],
    include_hand_end=config.include_hand_end,
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


action_ind = 2
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
# Study positional embeddings
QK_pos = einsum(
    model.pos_embed_copy.W_pos,
    W_QK,
    model.pos_embed_copy.W_pos,
    "pq dmq, h dmq dmk, pk dmk -> h pq pk",
)

QK_pos_tri = QK_pos.cpu().numpy()
for head in range(QK_pos_tri.shape[0]):
    tmp = QK_pos_tri[head, :, :]
    tmp[np.triu_indices(QK_pos_tri.shape[1], 1)] = -np.inf

# Role of each position in the cycle
POS_CYCLE = np.array(
    [
        "action / BOG",
        "result / SCORE",
        "action_0",
        "result_0",
        "action_1",
        "result_1",
        "hand_0",
        "hand_1",
        "hand_2",
        "hand_3",
        "hand_4",
        "hand_5",
        "hand_6",
        "hand_7",
    ]
)

if config.include_hand_end:
    POS_CYCLE = np.append(POS_CYCLE, "EOH")

# For each head, iterate through destination positions relevant to
# action predictions, and show the top-K source positions, sorted by their
# QK_pos_tri value
K = 5
top_src_pos_list = []
for head in range(QK_pos_tri.shape[0]):
    for action_ind, dst_pos in enumerate(
        range(first_action_pos - 1, QK_pos_tri.shape[1], action_pos_step)
    ):
        # Get the top-K source positions
        sort_inds = QK_pos_tri[head, dst_pos, :].argsort()[::-1]
        top_k_src_pos = sort_inds[:K]
        # Get the QK_pos values for the top-K source positions
        top_k_qk_pos = QK_pos_tri[head, dst_pos, top_k_src_pos]
        # Put this into a DataFrame
        top_src_pos_list.append(
            pd.DataFrame(
                {
                    "head": head,
                    "dst_pos": dst_pos,
                    "src_pos": top_k_src_pos,
                    "qk_val": top_k_qk_pos,
                    "action_ind": action_ind,
                }
            )
        )
top_src_pos_df = pd.concat(top_src_pos_list).set_index(["head", "action_ind"])
# Add a column for the dst and src position names take from POS_CYCLE
top_src_pos_df["dst_pos_next_name"] = POS_CYCLE[
    np.mod(top_src_pos_df.dst_pos + 1, len(POS_CYCLE))
]
top_src_pos_df["src_pos_name"] = POS_CYCLE[
    np.mod(top_src_pos_df.src_pos, len(POS_CYCLE))
]
# Add column for the turn index of the source positions
top_src_pos_df["src_pos_turn"] = (
    top_src_pos_df.src_pos - first_action_pos
) // action_pos_step + 1

NUM_ACTIONS_PLOT = 5
for head in range(QK_pos_tri.shape[0]):
    last_action_pos = (
        first_action_pos + (NUM_ACTIONS_PLOT - 1) * action_pos_step
    )
    fig = px.imshow(
        QK_pos_tri[
            head,
            (first_action_pos - 1) : last_action_pos : action_pos_step,
            :last_action_pos,
        ],
        color_continuous_scale="piyg",
        color_continuous_midpoint=0,
        title=f"QK_pos, head: {head}",
    )
    # Update x/y axis labels
    fig.update_xaxes(
        ticktext=np.tile(POS_CYCLE, NUM_ACTIONS_PLOT),
        tickvals=np.arange(len(POS_CYCLE) * NUM_ACTIONS_PLOT),
        tickangle=-45,
        title_text="Source position",
    )
    fig.update_yaxes(
        title_text="Action index",
    )
    fig.show()
    print(
        top_src_pos_df.loc[
            (head, slice(NUM_ACTIONS_PLOT)),
            ["src_pos_name", "src_pos_turn", "qk_val"],
        ]
    )


# QK_pos_plot = QK_pos[1].cpu().numpy()
# QK_pos_plot[np.triu_indices(QK_pos_plot.shape[0], -1)] = np.nan
# QK_pos_plot = QK_pos_plot[(first_action_pos - 1) :: action_pos_step, :]
# px.imshow(
#     QK_pos_plot,
#     color_continuous_scale="piyg",
#     color_continuous_midpoint=0,
#     aspect="auto",
# ).show()


# %%
# Explore the embeddings a bit

# First, PCA the positional embedding weights to get to a reasonable
# number of dimensions
pca_pos = PCA(n_components=15)
W_pos_pca = pca_pos.fit_transform(model.pos_embed_copy.W_pos.cpu().numpy())
print(pca_pos.explained_variance_ratio_)

# Now try t-SNE
W_pos_tsne = TSNE(n_components=2, random_state=3).fit_transform(W_pos_pca)
W_pos_tsne_df = pd.DataFrame(W_pos_tsne, columns=["tsne_0", "tsne_1"])
W_pos_tsne_df["pos_cycle_idx"] = [
    idx % len(POS_CYCLE) for idx in range(W_pos_pca.shape[0])
]
W_pos_tsne_df["pos_cycle_name"] = POS_CYCLE[W_pos_tsne_df.pos_cycle_idx]
W_pos_tsne_df["turn"] = W_pos_tsne_df.index // len(POS_CYCLE)
W_pos_tsne_df["pos_cycle_compr_name"] = W_pos_tsne_df["pos_cycle_name"].apply(
    lambda nm: "hand" if nm.startswith("hand") else nm
)

px.scatter(
    W_pos_tsne_df,
    x="tsne_0",
    y="tsne_1",
    color=W_pos_tsne_df["turn"].astype(str),
    hover_data=["pos_cycle_name"],
).show()

px.scatter(
    W_pos_tsne_df,
    x="tsne_0",
    y="tsne_1",
    color="pos_cycle_compr_name",
    hover_data=["turn"],
).show()

# %%
# What about embedding cosine sims?
csim_pos_tokens = sklearn.metrics.pairwise.cosine_similarity(
    model.pos_embed_copy.W_pos.cpu().numpy(),
    model.token_embed.W_E.cpu().numpy(),
)


# %%
# Load some example game data
NUM_GAMES = 1000


def game_filter(summary_lists):
    """Filter out games that don't match criteria"""
    inds = []
    for idx, player_inds in enumerate(summary_lists["player_indices"]):
        if (
            all(
                [
                    player_ind in config.test_player_inds
                    for player_ind in player_inds
                ]
            )
            and len(inds) < NUM_GAMES
        ):
            inds.append(idx)
    return inds


game_data = cheat_utils.load_game_data(
    dataset_folder=config.dataset_folder,
    sequence_mode=config.sequence_mode,
    game_filter=game_filter,
    device=model.cfg.device,
    include_hand_end=config.include_hand_end,
)


# %%
# Calculate the equivalent QK and OV matrices based on the combined
# token and position embeddings, of which only certain combinations are
# possible.

# Goal score as a baseline for interpreting the model
INTERP_GOAL_SCORE = 5

# Iterate over positions, and for each position, iterate over the
# possible tokens at that position, storing each possible (pos, token)
# tuple
TOKENS_BY_POS_CYCLE = {
    "action / BOG": ["BOG"]
    + token_props[token_props["type"] == "player_action"].index.to_list(),
    "result / SCORE": ["SCORE"]
    + token_props[token_props["type"] == "player_result"].index.to_list(),
    "action_0": ["PAD", "EOG"]
    + token_props[
        (token_props["type"] == "other_action") & (token_props["player"] == 0)
    ].index.to_list(),
    "result_0": ["PAD"]
    + token_props[
        (token_props["type"] == "other_result") & (token_props["player"] == 0)
    ].index.to_list(),
    "action_1": ["PAD"]
    + token_props[
        (token_props["type"] == "other_action") & (token_props["player"] == 1)
    ].index.to_list(),
    "result_1": ["PAD"]
    + token_props[
        (token_props["type"] == "other_result") & (token_props["player"] == 1)
    ].index.to_list(),
    "hand_0": token_props[
        (token_props["type"] == "hand") & (token_props["hand_rank"] == 0)
    ].index.to_list(),
    "hand_1": token_props[
        (token_props["type"] == "hand") & (token_props["hand_rank"] == 1)
    ].index.to_list(),
    "hand_2": token_props[
        (token_props["type"] == "hand") & (token_props["hand_rank"] == 2)
    ].index.to_list(),
    "hand_3": token_props[
        (token_props["type"] == "hand") & (token_props["hand_rank"] == 3)
    ].index.to_list(),
    "hand_4": token_props[
        (token_props["type"] == "hand") & (token_props["hand_rank"] == 4)
    ].index.to_list(),
    "hand_5": token_props[
        (token_props["type"] == "hand") & (token_props["hand_rank"] == 5)
    ].index.to_list(),
    "hand_6": token_props[
        (token_props["type"] == "hand") & (token_props["hand_rank"] == 6)
    ].index.to_list(),
    "hand_7": token_props[
        (token_props["type"] == "hand") & (token_props["hand_rank"] == 7)
    ].index.to_list(),
}

if config.include_hand_end:
    TOKENS_BY_POS_CYCLE["EOH"] = ["EOH"]

src_pos_token_list = []
dst_pos_token_list = []
for pos in range(model.pos_embed_copy.W_pos.shape[0]):
    pos_cycle_name = POS_CYCLE[pos % len(POS_CYCLE)]
    tokens_this = TOKENS_BY_POS_CYCLE[pos_cycle_name]
    pos_token_this = [(pos, vocab[token]) for token in tokens_this]
    src_pos_token_list.extend(pos_token_this)
    if (pos - first_action_pos) % action_pos_step == (action_pos_step - 1):
        dst_pos_token_list.extend(pos_token_this)

# Extract the position and token inds as vectors
src_pos, src_tok = t.tensor(src_pos_token_list).T.to(model.cfg.device)
dst_pos, dst_tok = t.tensor(dst_pos_token_list).T.to(model.cfg.device)

# Add the goal score embed to the position embedding
pos_embed_score = model.pos_embed_copy.W_pos.clone()
pos_embed_score[(POS_CYCLE == "result / SCORE").nonzero()[0][0], :] += (
    model.score_embed.b_S + model.score_embed.W_S * INTERP_GOAL_SCORE
)


def apply_layernorm(x):
    """Center and scale the last dimension of the input tensor."""
    x = x - x.mean(axis=-1, keepdim=True)
    scale = (x.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
    x = x / scale
    return x


# Calculate the combined embeddings of each possible source and
# destination token/pos, including a residual stream element that is
# always one to allow folding of the biases into the weights
def get_combined_embeds(
    pos, tok, pos_embed, token_embed, qkv_mode, double_embeds
):
    """Get the combined embeddings of the provided positions and tokens,
    including a residual stream element that is always one to allow
    folding of the biases into the weights. Optionally apply centering
    and scaling to simulate layernorm, and add dimension of 1s for bias
    folding. Both of these happen in "QKV mode".
    """
    embed = pos_embed[pos, :] + token_embed[tok, :]
    if double_embeds:
        embed *= 2
    if not qkv_mode:
        return embed
    # Center/scale
    embed = apply_layernorm(embed)
    # Add 1s dimension for bias folding
    return t.cat(
        [
            embed,
            t.ones(len(pos)).to(pos_embed.device)[:, None],
        ],
        dim=1,
    )


do_double_embeds = isinstance(model.pos_embed, IdentityVar)

embed_comb_qkv_src = get_combined_embeds(
    src_pos,
    src_tok,
    pos_embed_score,
    model.token_embed.W_E,
    qkv_mode=True,
    double_embeds=do_double_embeds,
)
embed_comb_qkv_dst = get_combined_embeds(
    dst_pos,
    dst_tok,
    pos_embed_score,
    model.token_embed.W_E,
    qkv_mode=True,
    double_embeds=do_double_embeds,
)
embed_comb_resid = get_combined_embeds(
    src_pos,
    src_tok,
    pos_embed_score,
    model.token_embed.W_E,
    qkv_mode=False,
    double_embeds=do_double_embeds,
)

# Calculate the QK weight matrix with biases folded in
W_QKb = einsum(
    t.cat(
        [model.blocks[0].attn.W_Q, model.blocks[0].attn.b_Q[:, None, :]], dim=1
    ),
    t.cat(
        [model.blocks[0].attn.W_K, model.blocks[0].attn.b_K[:, None, :]], dim=1
    ),
    "h dmq dh, h dmk dh -> h dmq dmk",
)

# Calculate the combined QK matrix, including the effect of the
# attention scaling before the softmax
QK_comb = (
    einsum(
        embed_comb_qkv_dst,
        W_QKb,
        embed_comb_qkv_src,
        "ptq dmq, h dmq dmk, ptk dmk -> h ptq ptk",
    )
    / model.blocks[0].attn.attn_scale
)
# Enforce causality
QK_comb = t.where(
    (dst_pos[:, None] >= src_pos[None, :])[None, :, :],
    QK_comb,
    t.tensor(-np.inf).to(model.cfg.device),
)
# Exponentiate to capture the first part of the softmax attention
# operation
QK_comb_exp = t.exp(QK_comb)

# Calculate the OV weight matrix with value biases folded in
W_OVb = einsum(
    model.blocks[0].attn.W_O,
    t.cat(
        [model.blocks[0].attn.W_V, model.blocks[0].attn.b_V[:, None, :]], dim=1
    ),
    "h dh dmo, h dmv dh -> h dmo dmv",
)

# # Calculate the combined OV matrix
# # Actually, don't use this, as it ignores layernorm which I don't want todo
# OV_comb = einsum(
#     model.unembed.W_U,
#     W_OVb,
#     embed_comb_qkv_src,
#     "dmo da, h dmo dmv, ptv dmv -> h da ptv",
# )
# # Calculate the combined action biases
# # Actually, don't use these, they ignore layernorm
# action_biases = (
#     einsum(model.unembed.W_U, model.blocks[0].attn.b_O, "dmo da, dmo -> da")
#     + model.unembed.b_U
# )

# OV without unembed
OV_comb = einsum(
    W_OVb,
    embed_comb_qkv_src,
    "h dmo dmv, ptv dmv -> h dmo ptv",
)

# Mapping from position and token to index in the combined embedding
pos_tok_to_src_embed_ind = {
    (pos, tok): idx for idx, (pos, tok) in enumerate(src_pos_token_list)
}
pos_tok_to_dst_embed_ind = {
    (pos, tok): idx for idx, (pos, tok) in enumerate(dst_pos_token_list)
}


def get_embed_inds(positions, tokens, pos_tok_to_ind):
    """Get the combined embedding indices for the provided positions and
    tokens."""
    return (
        t.tensor(
            [
                pos_tok_to_ind[(pos.item(), tok.item())]
                for pos, tok in zip(positions.flatten(), tokens.flatten())
            ]
        )
        .to(model.cfg.device)
        .reshape(positions.shape)
    )


def get_model_values(positions, tokens, embed, QK, OV, b_O, W_U, b_U):
    """Calculate the action logits and other stuff for a given set of tokens
    and positions. The destination is assumed to be the final token/pos.
    Layernorm and unembed are applied to the final residual stream value."""
    # Embedding indices
    embed_inds = get_embed_inds(positions, tokens, pos_tok_to_src_embed_ind)
    dst_ind = get_embed_inds(
        positions[:, [-1]], tokens[:, [-1]], pos_tok_to_dst_embed_ind
    )
    # Select specific chuncks of the QK, OV and embed tensors given
    # these specific embedding indices
    QK_this = rearrange(QK[:, dst_ind, embed_inds], "h b ptv -> b h ptv")
    OV_this = rearrange(OV[:, :, embed_inds], "h dmo b ptv -> b h ptv dmo")
    embed_this = embed[embed_inds, :]
    # Attention pattern matrix
    # pattern = t.softmax(QK_this, dim=-1)
    pattern = QK_this / QK_this.sum(dim=-1, keepdim=True)
    # Attention output
    attn_out = (
        einsum(pattern, OV_this, "b h pts, b h pts dmo -> b dmo")
        + b_O[None, :]
    )
    # Final residual stream value
    resid_final = embed_this[:, -1, :] + attn_out
    # LayerNorm
    resid_final = apply_layernorm(resid_final)
    # Unembed
    logits = (
        einsum(
            W_U,
            resid_final,
            "dmo da, b dmo -> b da",
        )
        + b_U
    )
    return {
        "embed_inds": embed_inds,
        "embed": embed_this,
        "attn_scores_exp": QK_this,
        "OV_this": OV_this,
        "pattern": pattern,
        "attn_out": attn_out,
        "resid_final": resid_final,
        "logits": logits,
    }


# Test the resid final calculation
model_vals = get_model_values(
    positions=repeat(
        t.arange(0, first_action_pos).to(model.cfg.device), "p -> b p", b=3
    ),
    tokens=game_data.game_data.tokens[:3, :first_action_pos],
    embed=embed_comb_resid,
    QK=QK_comb_exp,
    OV=OV_comb,
    b_O=model.blocks[0].attn.b_O,
    W_U=model.unembed.W_U,
    b_U=model.unembed.b_U,
)

logits, cache = model.run_with_cache(
    tokens=game_data.game_data.tokens[:3, :first_action_pos],
    scores=t.tensor([INTERP_GOAL_SCORE]).to(model.cfg.device),
)

print(t.allclose(model_vals["logits"], logits[:, 0, :], atol=1e-4))

# Show the exponentiated QK values for the first action
action_ind = 0
max_ind = (
    (src_pos < (first_action_pos + action_ind * action_pos_step)).sum().item()
)
plot_df = pd.DataFrame(
    {
        "QK_exp": QK_comb_exp[0, action_ind, :].cpu().numpy(),
        "token": np.array([vocab_str[tok.item()] for tok in src_tok.cpu()]),
        "pos": src_pos.cpu().numpy(),
    }
).iloc[:max_ind]
px.scatter(
    plot_df,
    x=plot_df.index,
    y="QK_exp",
    color=plot_df["pos"].astype(str),
    hover_data=["token"],
).show()


# Function to get the logits for a given set of tokens, specifying
# various arguments to the get_logits function
def viz_sequence(token_strs):
    """Wrapper around get_logits to visualize the model in various ways
    given a sequence of tokens."""
    # Convert the token strs to positions and tokens
    positions = t.arange(len(token_strs)).to(model.cfg.device)[None, :]
    tokens = t.tensor([vocab[token_str] for token_str in token_strs]).to(
        model.cfg.device
    )[None, :]
    # Run the unwrapped model to get various specific quantities
    model_vals = get_model_values(
        positions=positions,
        tokens=tokens,
        embed=embed_comb_resid,
        QK=QK_comb_exp,
        OV=OV_comb,
        b_O=model.blocks[0].attn.b_O,
        W_U=model.unembed.W_U,
        b_U=model.unembed.b_U,
    )
    # Plot the attention scores
    plot_df = pd.DataFrame(
        {
            "attn_scores_exp": model_vals["attn_scores_exp"][0, 0, :]
            .cpu()
            .numpy(),
            "pattern": model_vals["pattern"][0, 0, :].cpu().numpy(),
            "token": np.array(token_strs),
            "pos": np.arange(len(token_strs)),
        }
    )
    px.scatter(
        plot_df,
        x=plot_df.index,
        y="pattern",
        hover_data=["token", "pos"],
    ).show()
    # Calculate the attention-weighted OV values
    attn_OV = model_vals["pattern"][:, :, :, None] * model_vals["OV_this"]
    # Center each OV value (equivalent to the final layernorm centering)
    attn_OV_centered = attn_OV - attn_OV.mean(dim=-1, keepdim=True)
    # Unembed each attention-weighted OV value
    attn_OV_unembed = einsum(
        model.unembed.W_U,
        attn_OV_centered,
        "dmo da, b h ptv dmo -> b h ptv da",
    )
    px.imshow(attn_OV_unembed.squeeze().cpu().numpy()).show()
    # Final logits, softmaxed
    px.scatter(t.softmax(model_vals["logits"].cpu().squeeze(), dim=-1)).show()

    return model_vals


# Construct some example inputs to demonstrate various scenarios
# Start with a simple "can play" on first action
token_strs = [
    "BOG",
    "SCORE",
    "PAD",
    "PAD",
    "oa_1_c01",
    "ar_1_normal",
    "hand_0x0",
    "hand_0x1",
    "hand_1x2",
    "hand_0x3",
    "hand_0x4",
    "hand_0x5",
    "hand_0x6",
    "hand_0x7",
    "EOH",
]
model_vals = viz_sequence(token_strs)

# TODO: Think about the
# best way to visualize each token's contribution to the final output
# action logits, then what those final logits actually are.  Probably
# just the attention-weighted OV values are reasonable, then showing the
# final including layernorm, bias and log-softmax as well?

# ---------------------------------------------------------------------------

# %%
# Get some performance data
test_players = [
    players_all[idx] for idx in results_all["config"]["test_player_inds"]
]

test_game_config = cheat.CheatConfig(**vars(game_config))
test_game_config.penalize_wrong_played_card = True


goal_score = 5
margins_list = []
cheat_rates_list = []
# for player in [results.model] + test_players:
player = model
for goal_score in tqdm(np.linspace(-4, 4, 20)):
    margins, cheat_rate = cheat_utils.run_test_games(
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


# %%
perf_data_score = pd.DataFrame(
    {
        "cheat_rate": cheat_rates,
        "win_rate": margins_df.set_index("goal_score")["win_rate"],
    }
).reset_index()


# %%
# Test a trivial intervention: use a hook to add an offset to the "pass"
# action logits
# Add a pytorch forward hook to the model to add an offset to the
# "pass" action logits
def make_pass_offset_hook(offset):
    def pass_offset_hook(module, input, output):
        output[:, :, vocab["a_pass"]] += offset

    return pass_offset_hook


win_rates = []
action_stats_list = []
for pass_offset in tqdm(np.linspace(-1, 5, 20)):
    # Register hook
    handle = model.register_forward_hook(make_pass_offset_hook(pass_offset))
    try:
        # Run games
        margins, action_types = cheat_utils.run_test_games(
            model=model,
            game_config=game_config,
            num_games=500,
            goal_score=5,
            max_turns=max(
                results_all["config"]["cached_game_data"]["summary"][
                    "turn_cnt"
                ]
            ),
            players=test_players,
            seed=0,
            show_progress=True,
            post_proc_func=get_action_types,
        )
    finally:
        # Clear all forward hooks
        handle.remove()

    win_rate = (margins > 0).mean()
    win_rates.append({"pass_offset": pass_offset, "win_rate": win_rate})
    actions_df_this = (
        pd.concat(action_types)
        .reset_index()
        .groupby(["can_play", "action_type"])
        .count()
    )
    actions_df_this["pass_offset"] = pass_offset
    action_stats_list.append(
        actions_df_this.reset_index().rename(columns={"index": "count"})
    )
win_rates_df = pd.DataFrame(win_rates)
action_types_df = pd.concat(action_stats_list)

# %%
action_stats_all = action_types_df.groupby(["pass_offset", "action_type"])[
    "count"
].sum()
cheat_counts = action_stats_all.loc[(slice(None), "cheat")]
all_counts = action_stats_all.groupby("pass_offset").sum()
cheat_rates = cheat_counts / all_counts
perf_data_offset = pd.DataFrame(
    {
        "cheat_rate": cheat_rates,
        "win_rate": win_rates_df.set_index("pass_offset")["win_rate"],
    }
).reset_index()


# %%
# Now merge together the two performance dataframes and plot cheat_rate
# vs win_rate with the different interventions color-coded
perf_data = pd.concat(
    [
        perf_data_score.assign(intervention="change prompt"),
        perf_data_offset.assign(intervention="increase pass prob"),
    ]
)
perf_data["honesty"] = 1.0 - perf_data.cheat_rate
fig = px.line(
    perf_data,
    x="honesty",
    y="win_rate",
    color="intervention",
    title="Effects of deception-reducing interventions",
)
fig.update_traces(mode="lines+markers")
fig.update_layout(
    width=500,
)
fig.update_layout(
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
    )
)
fig.show()

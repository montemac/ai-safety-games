# %%
# Imports, etc.

import pickle
import datetime
import glob
import os
import lzma

# TEMP
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from sortedcontainers import SortedList
from collections import defaultdict
import numpy as np
import pandas as pd
import torch as t
import torch.optim as optim
from torch.utils.data import random_split
from tqdm.auto import tqdm
import plotly.express as px
from transformer_lens import HookedTransformerConfig

from ai_safety_games import cheat, utils, training
from ai_safety_games.ScoreTransformer import (
    ScoreTransformer,
    ScoreTransformerConfig,
)

utils.enable_ipython_reload()

_ = t.set_grad_enabled(True)

# %%
# Load a dataset into memory
FOLDER = "../datasets/random_dataset_20230731T001342"
DEVICE = "cuda:0"

# Load config info
with open(os.path.join(FOLDER, "config.pkl"), "rb") as file:
    config_dict = pickle.load(file)
    game_config = cheat.CheatConfig(**config_dict["game.config"])
    game = cheat.CheatGame(game_config)

# Load summary info
with open(os.path.join(FOLDER, "summary.pkl"), "rb") as file:
    summary_lists = pickle.load(file)
    # TODO: explain where all these numbers come from :)
    # max_seq_len = (
    #     (max(summary_lists["turn_cnt"]) + 1)
    #     // game_config.num_players
    #     * (2 * game.config.num_players + game.config.num_ranks)
    # ) + 2


# Get the token vocab
vocab, player_action_vocab = game.get_token_vocab()
vocab_strs = {idx: tok for tok, idx in vocab.items()}

# Load games and convert to token tensors plus scores
tokens_list = []
scores_list = []
for game_idx, filename in enumerate(
    tqdm(sorted(glob.glob(os.path.join(FOLDER, "game_*.pkl")))[:200000])
):
    with lzma.open(filename, "rb") as file:
        game_results = pickle.load(file)
        # Rebuild state history
        state_history = [
            cheat.CheatState(**state_dict)
            for state_dict in game_results["state_history_list"]
        ]
        # Calculate scores from state history (there was a bug in scores
        # calculation before so don't use loaded scores)
        hand_sizes = np.array([len(hand) for hand in state_history[-1].hands])
        scores = -hand_sizes
        next_best_score = np.max(scores[scores != 0])
        scores[scores == 0] = -next_best_score
        # Get token sequences
        tokens_this = cheat.get_seqs_from_state_history(
            game=game, vocab=vocab, state_history=state_history
        )
        tokens_list.extend([row for row in tokens_this])
        scores_list.append(scores)

# Convert to single tokens tensor, scores tensor, and seq lengths tensor
# Use pad_sequence to pad to the max length of any game
tokens_all = t.nn.utils.rnn.pad_sequence(
    tokens_list, batch_first=True, padding_value=0
).to(DEVICE)
scores_all = t.tensor(np.array(scores_list).flatten(), dtype=t.float32).to(
    DEVICE
)
seq_lens_all = t.tensor([len(toks) for toks in tokens_list], dtype=t.int64).to(
    DEVICE
)

# Positions corresponding to actions
first_action_pos = (
    (game_config.num_players - 1) * 2 + 2 + game.config.num_ranks
)
action_pos_step = game_config.num_players * 2 + game.config.num_ranks

# Turn sequence lengths into a loss mask
loss_mask = t.zeros_like(tokens_all, dtype=t.float32)
for idx, seq_len in enumerate(seq_lens_all):
    loss_mask[idx, :seq_len] = 1
loss_mask = loss_mask[:, first_action_pos::action_pos_step]


# %%


# %%
# TEMP: hack to fix rtgs to use victory_margin
# for game_idx, filename in enumerate(
#     tqdm(sorted(glob.glob(os.path.join(FOLDER, "rsas_*.pkl"))))
# ):
#     with open(filename, "rb") as file:
#         reward, state, action = pickle.load(file)
#     if t.max(reward).item() == 0:
#         print("here")
#         print(reward[0, 0].item())
#         winning_player_idx = t.argmax(reward[:, 0]).item()
#         other_indices = [
#             idx for idx in range(reward.shape[0]) if idx != winning_player_idx
#         ]
#         best_other_score = t.max(reward[other_indices, 0]).item()
#         reward[winning_player_idx, :] = -best_other_score
#         # Save the modified reward tensor and other tensors back to disk
#         with open(filename, "wb") as file:
#             pickle.dump((reward, state, action), file)


# %%
# Training loop using library function
TRAINING_MINS = 20
BATCH_SIZE = 100
LOG_PERIOD = 10000
N_LAYERS = 8
D_MODEL = 128
D_HEAD = 16
ATTN_ONLY = False
LR = 0.001
WEIGHT_DECAY = 0.00
SEED = 0

# Split data into train and test sets
# TODO: probably a simpler way to do this
generator = t.Generator().manual_seed(SEED)
train_inds, test_inds = [
    t.tensor(subset)
    for subset in random_split(
        range(tokens_all.shape[0]), [0.8, 0.2], generator=generator
    )
]

# Initialize a simple test model
model = ScoreTransformer(
    cfg=HookedTransformerConfig(
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        d_head=D_HEAD,
        d_vocab=len(vocab),
        d_vocab_out=len(player_action_vocab),
        act_fn="relu",
        device=DEVICE,
        seed=SEED,
        n_ctx=seq_lens_all.max().item(),
        attn_only=ATTN_ONLY,
    ),
    st_cfg=ScoreTransformerConfig(
        first_action_pos=first_action_pos, action_pos_step=action_pos_step
    ),
)

# Standard test function
test_inds_small = test_inds[:1000]
test_func = training.make_standard_test_func(
    test_data=training.TrainingTensors(
        inputs=[tokens_all[test_inds_small], scores_all[test_inds_small]],
        output=tokens_all[test_inds_small, first_action_pos::action_pos_step]
        .detach()
        .clone(),
        loss_mask=loss_mask[test_inds_small],
    ),
    test_batch_size=BATCH_SIZE,
)

# Train!
results = training.train_custom_transformer(
    model=model,
    config=training.TrainingConfig(
        project_name="cheat",
        training_mins=TRAINING_MINS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        log_period=LOG_PERIOD,
        seed=SEED,
        save_results=True,
    ),
    training_data=training.TrainingTensors(
        inputs=[tokens_all[train_inds], scores_all[train_inds]],
        output=tokens_all[train_inds, first_action_pos::action_pos_step]
        .detach()
        .clone(),
        loss_mask=loss_mask[train_inds],
    ),
    test_func=test_func,
)

# %%
# Show training results
plot_df = results.results.melt(
    id_vars=["elapsed_time"],
    value_vars=["loss_train", "loss_test"],
    var_name="loss_type",
    value_name="loss",
)
px.line(
    plot_df,
    x="elapsed_time",
    y="loss",
    color="loss_type",
    title="Training loss",
).show()


# %%
# # Create a model and run the training loop

# # Hyperparams
# SEED = 0
# NUM_EPOCHS = 20

# # Initialize a simple test model
# model = models.DecisionTransformer(
#     models.DecisionTransformerConfig(
#         n_layers=4,
#         d_model=128,
#         d_head=16,
#         d_state=state_tensors[0].shape[-1],
#         d_action=D_ACTION,
#         act_fn="relu",
#         device=DEVICE,
#         seed=SEED,
#         n_timesteps=N_TIMESTEPS,
#         attn_only=False,
#     )
# )

# # Optimizer
# optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# # Run training loop.  For now, just make each game a batch, likely want
# # to do something more sophisticated later.
# training_results = []
# for epoch in tqdm(range(NUM_EPOCHS)):
#     # Run on training set
#     loss_total = 0
#     loss_cnt = 0
#     for batch_idx, (rtgs_batch, states_batch, actions_batch) in tqdm(
#         enumerate(zip(rewards_tensors, state_tensors, action_tensors)),
#         total=len(rewards_tensors),
#     ):
#         # Skip any games that are too long to fit in the context window
#         if rtgs_batch.shape[1] > N_TIMESTEPS:
#             continue

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Make sure actions are correct dtype
#         actions_batch = actions_batch.to(dtype=t.int64)

#         # Forward + backward + optimize
#         logits = model(
#             rtgs=rtgs_batch, states=states_batch, actions=actions_batch
#         )
#         loss = model.loss_fn(logits=logits, actions=actions_batch)
#         loss_total += loss.item() * rtgs_batch.shape[0]
#         loss_cnt += rtgs_batch.shape[0]
#         loss.backward()
#         optimizer.step()

#     # Calculate train loss for this epoch
#     loss_train = loss_total / loss_cnt

#     # Save stats
#     training_results.append(
#         {
#             "epoch": epoch,
#             "loss_train": loss_train,
#             # "loss_test_1d": loss_test_1d.item(),
#             # "loss_test_games": loss_test_games.item(),
#         }
#     )
#     print(f"Epoch: {epoch}, loss_train: {loss_train}")

# training_results = pd.DataFrame(training_results)
# # px.line(
# #     training_results.melt(
# #         id_vars=["epoch"],
# #         value_vars=["loss_train"],  # , "loss_test_1d", "loss_test_games"],
# #     ),
# #     x="epoch",
# #     y="value",
# #     color="variable",
# # ).show()

# # Create a timestamped output directory
# output_dir = os.path.join(
#     "cheat_train_output", datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
# )
# os.makedirs(output_dir, exist_ok=True)

# # Save the model and the training results
# with open(os.path.join(output_dir, "model.pkl"), "wb") as file:
#     pickle.dump(model, file)
# with open(os.path.join(output_dir, "training_results.pkl"), "wb") as file:
#     pickle.dump(training_results, file)

# # %%
# # Load results from a previous run
# output_dir = "cheat_train_output/20230710T080356"
# with open(os.path.join(output_dir, "training_results.pkl"), "rb") as file:
#     training_results = pickle.load(file)

# px.line(
#     training_results.melt(
#         id_vars=["epoch"],
#         value_vars=["loss_train"],
#     ),
#     x="epoch",
#     y="value",
#     color="variable",
# ).show()

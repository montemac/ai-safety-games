# %%
# Imports, etc.

import pickle
import datetime
import glob
import os

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

from ai_safety_games import cheat, utils, models, training

utils.enable_ipython_reload()

# %%
# Load a dataset into memory
FOLDER = "../datasets/random_dataset_20230707T111903"
DEVICE = "cuda:0"
N_TIMESTEPS = 66  # ~200 turns for a 3 player game

# Load config info
with open(os.path.join(FOLDER, "config.pkl"), "rb") as file:
    config_dict = pickle.load(file)
    game_config = cheat.CheatConfig(**config_dict["game.config"])
    players_all = [
        cheat.RandomCheatPlayer(player_config["probs_table"])
        for player_config in config_dict["players"]
    ]

max_turns = N_TIMESTEPS * game_config.num_players

# Load summary data
with open(os.path.join(FOLDER, "summary.pkl"), "rb") as file:
    summary_lists = pickle.load(file)

# Load all the RSA tensors into the GPU, in a list by game index
# Filter out any games that are too long to fit in the context window
rewards_tensors = []
state_tensors = []
action_tensors = []
game_indices = []
for game_idx, filename in enumerate(
    tqdm(sorted(glob.glob(os.path.join(FOLDER, "rsas_*.pkl"))))
):
    with open(filename, "rb") as file:
        reward, state, action = pickle.load(file)
    # Skip any games that are too long to fit in the context window
    if reward.shape[1] <= N_TIMESTEPS:
        game_indices.append(game_idx)
        rewards_tensors.append(reward.to(DEVICE))
        state_tensors.append(state.to(DEVICE))
        loaded_action = action.to(DEVICE)
        # If the action tensor has a third dimension, it's already one-hot
        # encoded, convert it back to contain indices instead
        if len(loaded_action.shape) == 3:
            loaded_action = t.argmax(loaded_action, dim=-1)
        action_tensors.append(loaded_action.to(dtype=t.int64))

# Join the tensors into a single tensor for each quantity
# TODO: this whole thing seems clunky and inefficient! Lots of copying
# Declare appropriate sized zero tensors first
rtgs = t.zeros(
    (len(rewards_tensors) * rewards_tensors[0].shape[0], N_TIMESTEPS),
    device=DEVICE,
)
states = t.zeros(
    (
        len(rewards_tensors) * rewards_tensors[0].shape[0],
        N_TIMESTEPS,
        state_tensors[0].shape[-1],
    ),
    device=DEVICE,
)
actions = t.zeros(
    (len(rewards_tensors) * rewards_tensors[0].shape[0], N_TIMESTEPS),
    device=DEVICE,
    dtype=t.int64,
)
batch_idx = 0
for rtg_tensor, state_tensor, action_tensor in zip(
    rewards_tensors, state_tensors, action_tensors
):
    batch_size_this = rtg_tensor.shape[0]
    rtgs[
        batch_idx : batch_idx + batch_size_this, : rtg_tensor.shape[1]
    ] = rtg_tensor
    states[
        batch_idx : batch_idx + batch_size_this, : state_tensor.shape[1]
    ] = state_tensor
    actions[
        batch_idx : batch_idx + batch_size_this, : action_tensor.shape[1]
    ] = action_tensor
    batch_idx += batch_size_this


# MAX_GAME_STEPS = max([reward.shape[1] for reward in rewards_tensors])
# TODO: This is a hack to get the max action index.  It should be
# obtained from the game config instead.
D_ACTION = max([action.max() + 1 for action in action_tensors])


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
TRAINING_MINS = 40
BATCH_SIZE = 100
LOG_PERIOD = 5000
N_LAYERS = 8
D_MODEL = 128
D_HEAD = 16
LR = 0.001
WEIGHT_DECAY = 0.01
SEED = 0

# Split data into train and test sets
# TODO: probably a simpler way to do this
generator = t.Generator().manual_seed(SEED)
train_inds, test_inds = [
    t.tensor(subset)
    for subset in random_split(
        range(rtgs.shape[0]), [0.8, 0.2], generator=generator
    )
]

# Initialize a simple test model
model = models.DecisionTransformer(
    models.DecisionTransformerConfig(
        n_layers=4,
        d_model=D_MODEL,
        d_head=D_HEAD,
        d_state=state_tensors[0].shape[-1],
        d_action=D_ACTION,
        act_fn="relu",
        device=DEVICE,
        seed=SEED,
        n_timesteps=N_TIMESTEPS,
        attn_only=False,
    )
)

# Train!
results = training.train_decision_transformer(
    model=model,
    config=training.TrainingConfig(
        project_name="cheat",
        training_mins=TRAINING_MINS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        log_period=LOG_PERIOD,
        seed=SEED,
    ),
    training_data=training.RSATensors(
        rtgs=rtgs[train_inds],
        states=states[train_inds],
        actions=actions[train_inds],
    ),
    # test_func=test_func,
)


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

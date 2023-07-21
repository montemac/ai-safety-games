# %%
# Imports, etc.

import os
import pickle

import torch as t
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import lovely_tensors as lt
import plotly.express as px
import time
import datetime

from ai_safety_games import models, utils, training

lt.monkey_patch()

utils.enable_ipython_reload()

t.set_printoptions(sci_mode=False)

# %%
# Create a simple test dataset of RSA tuples that implement a trivial
# game:
# - State is two bits, randomly chosen (no dependence on previous state)
# - Action is either 0 or 1
# - If the action == XOR(state), then timestep reward is 1, else 0
# - Games always last for exactly 10 timesteps

SEED = 0
DEVICE = "cuda:0"
GAME_STEPS = 20
NUM_GAMES = 500000
NUM_TEST_GAMES = 10000
NUM_GAMES_TO_KEEP_LOSS = 100

NUM_TRAIN_GAMES = NUM_GAMES - NUM_TEST_GAMES

rng = np.random.default_rng(seed=SEED)
# Initialize the training RSA tensors
rtgs = t.zeros((NUM_GAMES, GAME_STEPS), dtype=t.float32).to(DEVICE)
states = t.zeros((NUM_GAMES, GAME_STEPS, 2), dtype=t.float32).to(DEVICE)
actions = t.zeros((NUM_GAMES, GAME_STEPS), dtype=t.int64).to(DEVICE)

# Run the games to populate the training tensors
for game_idx in tqdm(range(NUM_GAMES)):
    # Initialize the game
    states_this = rng.integers(0, 2, size=(GAME_STEPS, 2))
    actions_this = rng.integers(0, 2, size=GAME_STEPS)
    rewards_this = np.zeros(GAME_STEPS)
    # Run the game
    for timestep in range(GAME_STEPS):
        if actions_this[timestep] == np.bitwise_xor(
            states_this[timestep, 0], states_this[timestep, 1]
        ):
            rewards_this[timestep] = 1
    # Calculate reward-to-go for each timestep
    rtgs_this = np.cumsum(rewards_this[::-1], axis=0)[::-1].copy()
    # print(state)
    # print(np.bitwise_xor(state[:, 0], state[:, 1]))
    # print(action)
    # print(reward)
    # print(reward_to_go)
    # Add the game to the dataset
    rtgs[game_idx] = t.tensor(rtgs_this, dtype=t.float32)
    states[game_idx] = t.tensor(states_this, dtype=t.float32)
    actions[game_idx] = t.tensor(actions_this, dtype=t.int64)

# Split games into training and test sets
rtgs_test = rtgs[:NUM_TEST_GAMES]
states_test = states[:NUM_TEST_GAMES]
actions_test = actions[:NUM_TEST_GAMES]
rtgs = rtgs[NUM_TEST_GAMES:]
states = states[NUM_TEST_GAMES:]
actions = actions[NUM_TEST_GAMES:]

# %%
# Create various test setups for evaluating the model
# TESTS = {
#     "final_pos_perms": training.DecisionTransformerTest(
#         data=training.RSATensors(
#             rtgs=t.ones((4, GAME_STEPS), dtype=t.float32).to(DEVICE),
#             states=t.cat(
#                 [
#                     t.zeros((4, GAME_STEPS - 1, 2), dtype=t.float32).to(
#                         DEVICE
#                     ),
#                     t.tensor(
#                         [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=t.float32
#                     )[:, None, :].to(DEVICE),
#                 ],
#                 dim=1,
#             ),
#             actions=t.cat(
#                 [
#                     t.ones((4, GAME_STEPS - 1), dtype=t.int64).to(DEVICE),
#                     t.tensor([0, 1, 1, 0], dtype=t.int64).to(DEVICE)[:, None],
#                 ],
#                 dim=1,
#             ),
#         ),
#         loss_mask=t.arange(GAME_STEPS).to(DEVICE) == (GAME_STEPS - 1),
#         reduction="mean_timestep",
#     ),
#     "perfect_games": training.DecisionTransformerTest(
#         data=training.RSATensors(
#             rtgs=t.arange(GAME_STEPS, 0, -1)
#             .to(DEVICE)
#             .expand([NUM_TEST_GAMES, -1]),
#             states=states_test,
#             actions=t.logical_xor(
#                 states_test[:, :, 0], states_test[:, :, 1]
#             ).to(dtype=t.int64),
#         ),
#     ),
# }

# Create a test set of trajectories where the final state varies across all
# perms, the final RTG is 1, and the preceeding states are all dummies
# (RTG 1, state 0, 0, action 1)
with t.no_grad():
    # 1D test, at final position
    rtgs_1d_test = t.ones((4, GAME_STEPS), dtype=t.float32).to(DEVICE)
    states_1d_test = t.zeros((4, GAME_STEPS, 2), dtype=t.float32).to(DEVICE)
    states_1d_test[:, -1, :] = t.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=t.float32
    ).to(DEVICE)
    actions_1d_test = t.ones((4, GAME_STEPS - 1), dtype=t.int64).to(DEVICE)
    next_actions_correct_1d_test = t.tensor([0, 1, 1, 0], dtype=t.int64).to(
        DEVICE
    )[:, None]

    # Test for correct actions at all positions
    rtgs_correct_test = (
        t.arange(GAME_STEPS, 0, -1).to(DEVICE).expand([NUM_TEST_GAMES, -1])
    )
    actions_correct_test = t.logical_xor(
        states_test[:NUM_TEST_GAMES, :, 0], states_test[:NUM_TEST_GAMES, :, 1]
    ).to(dtype=t.int64)


# Define test function
def test_func(
    model: models.DecisionTransformer,
    config: training.TrainingConfig,
    test_idx: int,
):
    """Test function for the XOR game. Does 1D "final position" tests,
    as well as a test for perfect score on random games."""
    # 1D final position
    final_logits = model(
        rtgs=rtgs_1d_test,
        states=states_1d_test,
        actions=actions_1d_test,
    )[:, [-1], :]
    loss_1d_test = model.loss_fn(
        logits=final_logits, actions=next_actions_correct_1d_test
    )
    # All positions
    # Get logits for all positions
    logits = model(
        rtgs=rtgs_correct_test,
        states=states_test,
        actions=actions_correct_test,
    )
    # Take argmax over actions to get predicted actions
    actions_pred = t.argmax(logits, dim=-1)
    # Calculate mean accuracy for each position
    accuracy_test = t.mean(
        (actions_pred == actions_correct_test).float(), dim=0
    )
    # Calculate mean loss for each position
    loss_test_full = model.loss_fn(
        logits=logits,
        actions=actions_correct_test,
        per_token=True,
    )
    loss_test = t.mean(
        loss_test_full,
        dim=0,
    )
    # Store full loss tensors for a small set of test games
    loss_test_full_keep = loss_test_full[:NUM_GAMES_TO_KEEP_LOSS]
    # Package them all up and return
    return {
        "loss_1d_test": loss_1d_test.item(),
        "accuracy_test": accuracy_test.detach().cpu().numpy(),
        "loss_test": loss_test.detach().cpu().numpy(),
        "loss_test_full_keep": loss_test_full_keep.detach().cpu().numpy(),
    }


# %%
# Train using library function

# Hyperparams
TRAINING_MINS = 5
BATCH_SIZE = 1000
LOG_PERIOD = 50000
D_MODEL = 64
D_HEAD = 8
LR = 0.001
WEIGHT_DECAY = 0.01

# Initialize a simple test model
model = models.DecisionTransformer(
    models.DecisionTransformerConfig(
        n_layers=1,
        d_model=D_MODEL,
        d_head=D_HEAD,
        d_state=2,
        d_action=2,
        act_fn="relu",
        device=DEVICE,
        seed=SEED,
        n_timesteps=GAME_STEPS,
        attn_only=True,
    )
)

results = training.train_decision_transformer(
    model=model,
    config=training.TrainingConfig(
        project_name="xor",
        training_mins=TRAINING_MINS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        log_period=LOG_PERIOD,
        seed=SEED,
    ),
    training_data=training.RSATensors(
        rtgs=rtgs,
        states=states,
        actions=actions,
    ),
    test_func=test_func,
)


# %%
# Test the model
model = results.model
training_results = results.results
config = results.config

training_results["num_training_seqs"] = (
    np.arange(len(training_results)) * config.log_period
)

px.line(training_results, x="num_training_seqs", y="loss_train").show()

# Put test loss and accuracy over positions and training examples into a
# single dataframe
test_perf_by_position = pd.concat(
    [
        pd.DataFrame(
            np.array(training_results[col].to_list()),
            index=training_results["num_training_seqs"],
            columns=pd.Series(np.arange(GAME_STEPS), name="timestep"),
        )
        for col in ["loss_test", "accuracy_test"]
    ],
    axis=1,
    keys=["loss", "accuracy"],
    names=["qty"],
)

plot_df = test_perf_by_position.reset_index().melt(
    id_vars=["num_training_seqs"]
)
plot_df["timestep"] = plot_df["timestep"].astype(int)
px.scatter(
    plot_df,
    x="num_training_seqs",
    y="value",
    color="timestep",
    facet_col="qty",
    title="Mean test loss/accuracy at positions over training, XOR, perfect RTG",
).show()


# px.line(test_loss_results["final_pos_perms"]).show()

# training_results["num_training_seqs"] = (
#     np.arange(len(training_results)) * LOG_PERIOD
# )

# plot_df = training_results.melt(
#     id_vars=["elapsed_time", "epoch", "num_training_seqs"],
#     value_vars=["loss_train"],
# )
# plot_df
# px.line(
#     plot_df,
#     x="num_training_seqs",
#     y="value",
#     color="variable",
#     title="Loss on various sequence sets over training, XOR",
# ).show()

# plot_df = (
#     pd.DataFrame(test_loss_results)
#     .stack()
#     .reset_index()
#     .set_axis(["test_idx", "timestep", "mean_test_loss"], axis=1)
# )
# plot_df["num_training_seqs"] = plot_df["test_idx"] * LOG_PERIOD
# px.scatter(
#     plot_df,
#     x="num_training_seqs",
#     y="mean_test_loss",
#     color="timestep",
#     title="Mean test loss at positions over training, XOR, perfect RTG",
# ).show()


# %%
# Train!

# Hyperparams
# NUM_EPOCHS = 50
TRAINING_MINS = 15
BATCH_SIZE = 1000
NUM_GAMES_TO_USE = num_train_games
NUM_GAMES_TO_KEEP_LOSS = 100
LOG_PERIOD = 50000
D_MODEL = 64
D_HEAD = 8
LR = 0.001
WEIGHT_DECAY = 0.01

# Initialize a simple test model
model = models.DecisionTransformer(
    models.DecisionTransformerConfig(
        n_layers=1,
        d_model=D_MODEL,
        d_head=D_HEAD,
        d_state=2,
        d_action=2,
        act_fn="relu",
        device=DEVICE,
        seed=SEED,
        n_timesteps=GAME_STEPS,
        attn_only=True,
    )
)

# Train the model
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

dataset = TensorDataset(
    rtgs[:NUM_GAMES_TO_USE],
    states[:NUM_GAMES_TO_USE],
    actions[:NUM_GAMES_TO_USE],
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

training_results = []
test_accuracy_results = []
test_loss_results = []
test_loss_full_results = []
start_time = time.time()
elapsed_mins = 0
epoch = 0
since_last_log = 0
loss_total = 0
loss_cnt = 0
progress_bar = tqdm(total=TRAINING_MINS)
# for epoch in tqdm(range(NUM_EPOCHS)):
while elapsed_mins < TRAINING_MINS:
    for batch_idx, (rtgs_batch, states_batch, actions_batch) in enumerate(
        dataloader
    ):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        logits = model(
            rtgs=rtgs_batch, states=states_batch, actions=actions_batch
        )
        loss = model.loss_fn(logits=logits, actions=actions_batch)
        loss_total += loss.item() * rtgs_batch.shape[0]
        loss_cnt += rtgs_batch.shape[0]
        loss.backward()
        optimizer.step()

        # Calculate elapsed time and update progress bar
        elapsed_mins = (time.time() - start_time) / 60
        progress_bar.update(elapsed_mins - progress_bar.n)

        # Determine whether to log
        since_last_log += rtgs_batch.shape[0]
        if since_last_log >= LOG_PERIOD:
            # Run on test sets
            with t.no_grad():
                # 1D final position
                final_logits = model(
                    rtgs=rtgs_1d_test,
                    states=states_1d_test,
                    actions=actions_1d_test,
                )[:, [-1], :]
                loss_1d_test = model.loss_fn(
                    logits=final_logits, actions=next_actions_correct_1d_test
                )
                # All positions
                # Get logits for all positions
                logits = model(
                    rtgs=rtgs_correct_test,
                    states=states_test,
                    actions=actions_correct_test,
                )
                # Take argmax over actions to get predicted actions
                actions_pred = t.argmax(logits, dim=-1)
                # Calculate mean accuracy for each position
                accuracy_test = t.mean(
                    (actions_pred == actions_correct_test).float(), dim=0
                )
                # Calculate mean loss for each position
                loss_test_full = model.loss_fn(
                    logits=logits,
                    actions=actions_correct_test,
                    per_token=True,
                )
                loss_test = t.mean(
                    loss_test_full,
                    dim=0,
                )
                # Store full loss tensors for a small set of test games
                test_loss_full_keep = loss_test_full[:NUM_GAMES_TO_KEEP_LOSS]

            # Update training loss
            loss_train = loss_total / loss_cnt
            loss_total = 0
            loss_cnt = 0

            # Store stats
            training_results.append(
                {
                    "elapsed_time": elapsed_mins,
                    "epoch": epoch,
                    "loss_train": loss_train,
                    "loss_test_1d": loss_1d_test.item(),
                    "loss_test_games": t.mean(loss_test).item(),
                }
            )
            test_accuracy_results.append(accuracy_test.detach().cpu().numpy())
            test_loss_results.append(loss_test.detach().cpu().numpy())
            test_loss_full_results.append(
                test_loss_full_keep.detach().cpu().numpy()
            )

            since_last_log = 0

    epoch += 1

training_results = pd.DataFrame(training_results)
test_accuracy_results = np.array(test_accuracy_results)
test_loss_results = np.array(test_loss_results)
test_loss_full_results = np.array(test_loss_full_results)

# Create a timestamped output directory
output_dir = os.path.join(
    "xor_output", datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
)
os.makedirs(output_dir, exist_ok=True)

# Save the model and the training results in a pickled dictionary
with open(os.path.join(output_dir, "results.pkl"), "wb") as file:
    pickle.dump(
        {
            "model": model,
            "training_results": training_results,
            "test_accuracy_results": test_accuracy_results,
            "test_loss_results": test_loss_results,
            "test_loss_full_results": test_loss_full_results,
        },
        file,
    )


# %%
# Load a pre-trained model and training results
output_dir = "xor_output/20230710T115945"
with open(os.path.join(output_dir, "results.pkl"), "rb") as file:
    results = pickle.load(file)
model = results["model"]
training_results = results["training_results"]
test_accuracy_results = results["test_accuracy_results"]
test_loss_results = results["test_loss_results"]

training_results["num_training_seqs"] = (
    np.arange(len(training_results)) * LOG_PERIOD
)

plot_df = training_results.melt(
    id_vars=["elapsed_time", "epoch", "num_training_seqs"],
    value_vars=["loss_train", "loss_test_1d", "loss_test_games"],
)
plot_df
px.line(
    plot_df,
    x="num_training_seqs",
    y="value",
    color="variable",
    title="Loss on various sequence sets over training, XOR",
).show()

# plot_df = (
#     pd.DataFrame(test_loss_results)
#     .stack()
#     .reset_index()
#     .set_axis(["test_idx", "timestep", "mean_test_loss"], axis=1)
# )
# plot_df["num_training_seqs"] = plot_df["test_idx"] * LOG_PERIOD
# px.scatter(
#     plot_df,
#     x="num_training_seqs",
#     y="mean_test_loss",
#     color="timestep",
#     title="Mean test loss at positions over training, XOR, perfect RTG",
# ).show()


# %%
# Test the model more thoroughly

# with t.no_grad():
#     logits = model(
#         rtgs=rtgs_all_correct,
#         states=states[:NUM_TEST_GAMES],
#         actions=actions_correct,
#     )
#     dist = t.distributions.categorical.Categorical(logits=logits.squeeze())

# prob_correct_action = dist.probs.gather(
#     dim=-1, index=actions_correct[..., None]
# )
# px.line(prob_correct_action.mean(dim=0).cpu().numpy()).show()

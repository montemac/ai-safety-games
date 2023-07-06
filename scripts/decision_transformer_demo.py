# %%
# Imports, etc.

import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import lovely_tensors as lt
import plotly.express as px

from ai_safety_games import models, utils

lt.monkey_patch()

utils.enable_ipython_reload()

torch.set_printoptions(sci_mode=False)

# %%
# Create a simple test dataset of RSA tuples that implement a trivial
# game:
# - State is two bits, randomly chosen (no dependence on previous state)
# - Action is either 0 or 1
# - If the action == XOR(state), then timestep reward is 1, else 0
# - Games always last for exactly 10 timesteps

SEED = 0
DEVICE = "cuda:0"
GAME_STEPS = 10
NUM_GAMES = 500000
NUM_TEST_GAMES = 100


rng = np.random.default_rng(seed=SEED)
# Initialize the training RSA tensors
rtgs = torch.zeros((NUM_GAMES, GAME_STEPS), dtype=torch.float32).to(DEVICE)
states = torch.zeros((NUM_GAMES, GAME_STEPS, 2), dtype=torch.float32).to(
    DEVICE
)
actions = torch.zeros((NUM_GAMES, GAME_STEPS), dtype=torch.int64).to(DEVICE)

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
    rtgs[game_idx] = torch.tensor(rtgs_this, dtype=torch.float32)
    states[game_idx] = torch.tensor(states_this, dtype=torch.float32)
    actions[game_idx] = torch.tensor(actions_this, dtype=torch.int64)

# Create a test set of trajectories where the final state varies across all
# perms, the final RTG is 1, and the preceeding states are all dummies
# (RTG 1, state 0, 0, action 1)
with torch.no_grad():
    # 1D test, at final position
    rtgs_test = torch.ones((4, GAME_STEPS), dtype=torch.float32).to(DEVICE)
    states_test = torch.zeros((4, GAME_STEPS, 2), dtype=torch.float32).to(
        DEVICE
    )
    states_test[:, -1, :] = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32
    ).to(DEVICE)
    actions_test = torch.ones((4, GAME_STEPS - 1), dtype=torch.int64).to(
        DEVICE
    )
    next_actions_correct_test = torch.tensor(
        [0, 1, 1, 0], dtype=torch.int64
    ).to(DEVICE)[:, None]

    # Test for correct actions at all positions
    rtgs_all_correct = (
        torch.arange(GAME_STEPS, 0, -1).to(DEVICE).expand([NUM_TEST_GAMES, -1])
    )
    actions_correct = torch.logical_xor(
        states[:NUM_TEST_GAMES, :, 0], states[:NUM_TEST_GAMES, :, 1]
    ).to(dtype=torch.int64)
    states_test_games = states[:NUM_TEST_GAMES].clone()


# %%
# Train!

# Hyperparams
NUM_EPOCHS = 50
BATCH_SIZE = 1000

# Initialize a simple test model
model = models.DecisionTransformer(
    models.DecisionTransformerConfig(
        n_layers=1,
        d_model=64,
        d_head=8,
        d_state=2,
        d_action=2,
        act_fn="relu",
        device=DEVICE,
        seed=SEED,
        n_timesteps=GAME_STEPS,
        attn_only=True,
    )
)

# Create a simple test dataset of RSA tuples that implement a trivial
# game:
# - State is two bits, randomly chosen (no dependence on previous state)
# - Action is either 0 or 1
# - If the action == XOR(state), then timestep reward is 1, else 0
# - Games always last for exactly 10 timesteps

rng = np.random.default_rng(seed=SEED)
# Initialize the training RSA tensors
rtgs = torch.zeros((NUM_GAMES, GAME_STEPS), dtype=torch.float32).to(DEVICE)
states = torch.zeros((NUM_GAMES, GAME_STEPS, 2), dtype=torch.float32).to(
    DEVICE
)
actions = torch.zeros((NUM_GAMES, GAME_STEPS), dtype=torch.int64).to(DEVICE)

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
    rtgs[game_idx] = torch.tensor(rtgs_this, dtype=torch.float32)
    states[game_idx] = torch.tensor(states_this, dtype=torch.float32)
    actions[game_idx] = torch.tensor(actions_this, dtype=torch.int64)

# Create a test set of trajectories where the final state varies across all
# perms, the final RTG is 1, and the preceeding states are all dummies
# (RTG 1, state 0, 0, action 1)
with torch.no_grad():
    # 1D test, at final position
    rtgs_test = torch.ones((4, GAME_STEPS), dtype=torch.float32).to(DEVICE)
    states_test = torch.zeros((4, GAME_STEPS, 2), dtype=torch.float32).to(
        DEVICE
    )
    states_test[:, -1, :] = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32
    ).to(DEVICE)
    actions_test = torch.ones((4, GAME_STEPS - 1), dtype=torch.int64).to(
        DEVICE
    )
    next_actions_correct_test = torch.tensor(
        [0, 1, 1, 0], dtype=torch.int64
    ).to(DEVICE)[:, None]

    # Test for correct actions at all positions
    rtgs_all_correct = (
        torch.arange(GAME_STEPS, 0, -1).to(DEVICE).expand([NUM_TEST_GAMES, -1])
    )
    actions_correct = torch.logical_xor(
        states[:NUM_TEST_GAMES, :, 0], states[:NUM_TEST_GAMES, :, 1]
    ).to(dtype=torch.int64)
    states_test_games = states[:NUM_TEST_GAMES].clone()

# Train the model
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

dataset = TensorDataset(rtgs, states, actions)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

training_results = []
for epoch in tqdm(range(NUM_EPOCHS)):
    # Run on test sets
    with torch.no_grad():
        # 1D final position
        final_logits = model(
            rtgs=rtgs_test,
            states=states_test,
            actions=actions_test,
        )[:, [-1], :]
        loss_test_1d = model.loss_fn(
            logits=final_logits, actions=next_actions_correct_test
        )
        # All positions
        logits = model(
            rtgs=rtgs_all_correct,
            states=states_test_games,
            actions=actions_correct,
        )
        loss_test_games = model.loss_fn(logits=logits, actions=actions_correct)

    # Run on training set, in batches, optimiizng
    loss_total = 0
    loss_cnt = 0
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
    loss_train = loss_total / loss_cnt

    # print statistics
    training_results.append(
        {
            "epoch": epoch,
            "loss_train": loss_train,
            "loss_test_1d": loss_test_1d.item(),
            "loss_test_games": loss_test_games.item(),
        }
    )

training_results = pd.DataFrame(training_results)
px.line(
    training_results.melt(
        id_vars=["epoch"],
        value_vars=["loss_train", "loss_test_1d", "loss_test_games"],
    ),
    x="epoch",
    y="value",
    color="variable",
).show()

# %%
# Test the model more thoroughly

with torch.no_grad():
    logits = model(
        rtgs=rtgs_all_correct,
        states=states[:NUM_TEST_GAMES],
        actions=actions_correct,
    )
    dist = torch.distributions.categorical.Categorical(logits=logits.squeeze())

prob_correct_action = dist.probs.gather(
    dim=-1, index=actions_correct[..., None]
)
px.line(prob_correct_action.mean(dim=0).cpu().numpy()).show()

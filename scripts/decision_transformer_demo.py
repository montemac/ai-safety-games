# %%
# Imports, etc.

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import lovely_tensors as lt
import plotly.express as px

from ai_safety_games import models, utils

lt.monkey_patch()

utils.enable_ipython_reload()

# %%
# Train!

# Hyperparams
SEED = 0
DEVICE = "cuda:0"
GAME_STEPS = 20
N_GAMES = 2000
NUM_EPOCHS = 300

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
rtgs = torch.zeros((N_GAMES, GAME_STEPS), dtype=torch.float32).to(DEVICE)
states = torch.zeros((N_GAMES, GAME_STEPS, 2), dtype=torch.float32).to(DEVICE)
actions = torch.zeros((N_GAMES, GAME_STEPS), dtype=torch.int64).to(DEVICE)

# Run the games to populate the training tensors
for game_idx in tqdm(range(N_GAMES)):
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
rtgs_test = torch.ones((4, GAME_STEPS), dtype=torch.float32).to(DEVICE)
states_test = torch.zeros((4, GAME_STEPS, 2), dtype=torch.float32).to(DEVICE)
states_test[:, -1, :] = torch.tensor(
    [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32
).to(DEVICE)
actions_test = torch.ones((4, GAME_STEPS - 1), dtype=torch.int64).to(DEVICE)
next_actions_correct_test = torch.tensor([0, 1, 1, 0], dtype=torch.int64).to(
    DEVICE
)[:, None]

# Train the model
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

training_results = []
for epoch in tqdm(range(NUM_EPOCHS)):
    # Run on test set
    with torch.no_grad():
        final_logits = model(
            rtgs=rtgs_test,
            states=states_test,
            actions=actions_test,
        )[:, [-1], :]
        loss_test = model.loss_fn(
            logits=final_logits, actions=next_actions_correct_test
        )

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward + backward + optimize
    logits = model(rtgs=rtgs, states=states, actions=actions)
    loss = model.loss_fn(logits=logits, actions=actions)
    loss.backward()
    optimizer.step()

    # print statistics
    training_results.append(
        {
            "epoch": epoch,
            "loss_train": loss.item(),
            "loss_test": loss_test.item(),
        }
    )

training_results = pd.DataFrame(training_results)
px.line(
    training_results.melt(
        id_vars=["epoch"], value_vars=["loss_train", "loss_test"]
    ),
    x="epoch",
    y="value",
    color="variable",
).show()

# %%
# Test the model

# Create some test trajectories where the final state varies across all
# perms, the final RTG is 1, and the preceeding states are all dummies
# (RTG 1, state 0, 0, action 1)
rtgs_test = torch.ones((4, GAME_STEPS), dtype=torch.float32).to(DEVICE)
states_test = torch.zeros((4, GAME_STEPS, 2), dtype=torch.float32).to(DEVICE)
states_test[:, -1, :] = torch.tensor(
    [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32
).to(DEVICE)
actions_test = torch.ones((4, GAME_STEPS - 1), dtype=torch.int64).to(DEVICE)

with torch.no_grad():
    final_logits = model(
        rtgs=rtgs_test,
        states=states_test,
        actions=actions_test,
    )[:, -1, :]
    dist = torch.distributions.categorical.Categorical(
        logits=final_logits.squeeze()
    )

dist.probs.p

"""Functions for training models."""

from collections import defaultdict
from dataclasses import dataclass, asdict
import datetime
import os
import pickle
import time
from typing import Union, Dict, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch as t
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from jaxtyping import Float32, Int64, Bool

from ai_safety_games import models


@dataclass
class TrainingConfig:
    """Configuration for training a model."""

    # Name of the training project
    project_name: str
    # Number of minutes to train for
    training_mins: float
    # Batch size
    batch_size: int
    # Learning rate
    lr: float
    # Weight decay
    weight_decay: float
    # Log period in number of batches
    log_period: int
    # Random seed, set everywhere
    seed: int
    # Whether to save results
    save_results: bool = True


@dataclass
class RSATensors:
    """Tensors for an RSA dataset."""

    rtgs: Float32[t.Tensor, "idx timestep"]
    states: Float32[t.Tensor, "idx timestep state"]
    actions: Int64[t.Tensor, "idx timestep"]


@dataclass
class DecisionTransformerTest:
    """Fields that define a test set for a decision transformer
    model."""

    data: RSATensors
    loss_mask: Optional[Bool[t.Tensor, "timestep"]] = None
    reduction: str = "mean_all"


@dataclass
class DecisionTransformerTrainingResults:
    """Results from training a decision transformer model."""

    config: TrainingConfig
    model: models.DecisionTransformer
    training_results: pd.DataFrame
    test_accuracy_results: np.ndarray
    test_loss_results: np.ndarray


def train_decision_transformer(
    model: models.DecisionTransformer,
    config: TrainingConfig,
    training_data: RSATensors,
    test_data: Union[
        DecisionTransformerTest, Dict[str, DecisionTransformerTest]
    ],
):
    """Train a decision transformer model given an RSA dataset. Assumes
    that the full dataset can fit in memory at once. Also assumes that
    all the test sets can be run in single batches (TODO: relax
    this)."""

    # TODO: add wandb logging

    # Set random seed everywhere
    t.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Make sure test data is a dict
    if not isinstance(test_data, dict):
        test_data = {"": test_data}

    # Create optimizer and dataloader
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    dataset = TensorDataset(
        training_data.rtgs,
        training_data.states,
        training_data.actions,
    )
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )

    # Initialize variables for logging
    training_results = []
    test_accuracy_results = defaultdict(list)
    test_loss_results = defaultdict(list)
    start_time = time.time()
    elapsed_mins = 0
    epoch = 0
    since_last_log = 0
    loss_total = 0
    loss_cnt = 0
    progress_bar = tqdm(total=config.training_mins)

    # Run the training loop
    # for epoch in tqdm(range(NUM_EPOCHS)):
    while elapsed_mins < config.training_mins:
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

            # Determine whether to run tests and log results
            since_last_log += rtgs_batch.shape[0]
            if since_last_log >= config.log_period:
                # Run on test sets
                with t.no_grad():
                    for test_name, test in test_data.items():
                        # Forward pass to get logits
                        actions = test.data.actions
                        logits = model(
                            rtgs=test.data.rtgs,
                            states=test.data.states,
                            actions=actions,
                        )
                        # Mask out logits and actions for timesteps
                        # where we don't care about the loss
                        if test.loss_mask is not None:
                            logits = logits[:, test.loss_mask, :]
                            actions = test.data.actions[:, test.loss_mask]
                        # Take argmax to get predicted actions, and
                        # calculate accuracy
                        actions_pred = t.argmax(logits, dim=-1)
                        accuracy = (actions_pred == actions).float()
                        # Calcualte loss
                        loss = model.loss_fn(
                            logits=logits,
                            actions=actions,
                            per_token=True,
                        )
                        # Reduce accuracy and loss according to specified method
                        if test.reduction == "mean_all":
                            accuracy = t.mean(accuracy)
                            loss = t.mean(loss)
                        elif test.reduction == "mean_timestep":
                            accuracy = t.mean(accuracy, dim=1)
                            loss = t.mean(loss, dim=1)
                        elif test.reduction == "mean_batch":
                            accuracy = t.mean(accuracy, dim=0)
                            loss = t.mean(loss, dim=0)
                        # Store results
                        test_accuracy_results[test_name].append(
                            accuracy.detach().cpu().numpy()
                        )
                        test_loss_results[test_name].append(
                            loss.detach().cpu().numpy()
                        )

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
                    }
                )

                since_last_log = 0

        epoch += 1

    training_results = pd.DataFrame(training_results)
    test_accuracy_results = {
        test_name: np.array(results)
        for test_name, results in test_accuracy_results.items()
    }
    test_loss_results = {
        test_name: np.array(results)
        for test_name, results in test_loss_results.items()
    }

    # Create a timestamped output directory
    output_dir = os.path.join(
        config.project_name + "_results",
        datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
    )
    os.makedirs(output_dir, exist_ok=True)

    # Store results
    final_results = DecisionTransformerTrainingResults(
        config=config,
        model=model,
        training_results=training_results,
        test_accuracy_results=test_accuracy_results,
        test_loss_results=test_loss_results,
    )

    # Save the model and the training results in a pickled dictionary
    if config.save_results:
        with open(os.path.join(output_dir, "results.pkl"), "wb") as file:
            pickle.dump(
                asdict(final_results),
                file,
            )

    return final_results

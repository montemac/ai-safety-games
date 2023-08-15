"""Functions for training models."""

from dataclasses import dataclass, asdict
import datetime
import os
import pickle
import time
from typing import Union, Dict, Optional, Callable, Any, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch as t
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from jaxtyping import Float32, Int64, Bool
from transformer_lens import HookedTransformer
import transformers


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    """Configuration for training a model."""

    project_name: str
    epochs: int
    batch_size: int
    lr: float
    lr_schedule: Optional[Tuple[str, Dict[str, Any]]] = None
    weight_decay: float
    # Log period in number of training examples
    log_period: int
    # Random seed, set everywhere
    seed: int
    save_results: bool = True


# Type hint for a dataset of tensors for training some kind of custom
# transformer
@dataclass
class TrainingTensors:
    """List of tensors that will be passed (in provided order) to
    forward call, plus a tensor of target outputs (e.g. actions) which
    might be a subset of the inputs."""

    inputs: list[t.Tensor]
    output: t.Tensor
    loss_mask: t.Tensor


@dataclass
class TrainingResults:
    """Results from training a decision transformer model."""

    config: TrainingConfig
    model: nn.Module
    results: pd.DataFrame


# Type hint for a test function that takes a module, training config,
# and test index, and returns a dictionary of test results
TestFunc = Callable[[nn.Module, TrainingConfig, int], Dict[str, Any]]


def make_standard_test_func(
    test_data: TrainingTensors, test_batch_size: int
) -> TestFunc:
    """Create a standard test function for a decision transformer model
    given a test dataset and batch size."""

    def test_func(model: nn.Module, config: TrainingConfig, test_idx: int):
        """Test function for a decision transformer model."""
        # Create a dataloader for the test set
        dataset = TensorDataset(
            *test_data.inputs, test_data.output, test_data.loss_mask
        )
        dataloader = DataLoader(
            dataset, batch_size=test_batch_size, shuffle=False
        )

        # Initialize variables for logging
        loss_total = 0
        loss_cnt = 0

        # Run the test loop
        with t.no_grad():
            for batch_idx, test_batch in enumerate(dataloader):
                # Split into inputs and outputs
                inputs_batch = test_batch[:-2]
                output_batch = test_batch[-2]
                loss_mask_batch = test_batch[-1]
                batch_size = output_batch.shape[0]

                # Forward pass
                logits = model(*inputs_batch)
                loss = model.loss_fn(
                    logits, output_batch, loss_mask=loss_mask_batch
                )
                loss_total += loss.item() * batch_size
                loss_cnt += batch_size

        # Update training loss
        loss_test = loss_total / loss_cnt

        # Save training results
        results_this = {
            "loss_test": loss_test,
        }

        return results_this

    return test_func


def train_custom_transformer(
    model: HookedTransformer,
    config: TrainingConfig,
    training_data: TrainingTensors,
    test_func: Optional[TestFunc] = None,
):
    """Train a decision transformer model given an RSA dataset. Assumes
    that the full dataset can fit in memory at once. Also assumes that
    all the test sets can be run in single batches (TODO: relax
    this)."""

    # TODO: add wandb logging

    # Set random seed everywhere
    t.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create optimizer and dataloader
    optimizer = t.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    dataset = TensorDataset(
        *training_data.inputs, training_data.output, training_data.loss_mask
    )
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )

    total_batches = config.epochs * len(dataloader)

    # TODO: add more complex LR scheduling, and go back to training for
    # a specific number of epochs (or batches) so that these schedules
    # can be used sensibly?
    # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules
    if config.lr_schedule is None:
        # Create dummy scheduler that just implements a constant LR
        scheduler = transformers.get_constant_schedule(optimizer)
    else:
        sched_name, sched_params = config.lr_schedule
        if sched_name == "cosine_with_warmup":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(
                    total_batches * sched_params["warmup_fraction"]
                ),
                num_training_steps=total_batches,
            )

    # Initialize variables for logging
    training_results = []
    start_time = time.time()
    elapsed_mins = 0
    # epoch = 0
    batch = 0
    since_last_log = 0
    loss_total = 0
    loss_cnt = 0
    # progress_bar = tqdm(total=config.training_mins)

    # Run the training loop
    with tqdm(total=total_batches) as progress_bar:
        for epoch in range(config.epochs):
            # while elapsed_mins < config.training_mins:
            for batch_idx, training_batch in enumerate(dataloader):
                # Split into inputs and outputs
                inputs_batch = training_batch[:-2]
                output_batch = training_batch[-2]
                loss_mask_batch = training_batch[-1]
                batch_size = output_batch.shape[0]

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                logits = model(*inputs_batch)
                loss = model.loss_fn(
                    logits, output_batch, loss_mask=loss_mask_batch
                )
                loss_total += loss.item() * batch_size
                loss_cnt += batch_size
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Calculate elapsed time and update progress bar
                elapsed_mins = (time.time() - start_time) / 60
                progress_bar.update(1)
                # progress_bar.update(elapsed_mins - progress_bar.n)
                # if elapsed_mins >= config.training_mins:
                #     progress_bar.close()
                #     break

                # Determine whether to run tests and log results
                since_last_log += batch_size
                if since_last_log >= config.log_period:
                    # Update training loss
                    loss_train = loss_total / loss_cnt
                    loss_total = 0
                    loss_cnt = 0

                    # Save training results
                    results_this = {
                        "elapsed_time": elapsed_mins,
                        "epoch": epoch,
                        "batch": batch,
                        "lr": scheduler.get_last_lr()[0],
                        "loss_train": loss_train,
                    }

                    # Run test func and save results, if provided
                    if test_func is not None:
                        with t.no_grad():
                            test_results = test_func(
                                model, config, len(training_results)
                            )
                            results_this.update(test_results)

                    # Store results
                    training_results.append(results_this)

                    since_last_log = 0

                batch += 1

            # epoch += 1

    training_results = pd.DataFrame(training_results)

    # Store results
    final_results = TrainingResults(
        config=config,
        model=model,
        results=training_results,
    )

    # Save the model and the training results in a pickled dictionary
    if config.save_results:
        # Create a timestamped output directory
        output_dir = os.path.join(
            config.project_name + "_results",
            datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
        )
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "results.pkl"), "wb") as file:
            pickle.dump(
                asdict(final_results),
                file,
            )

    return final_results

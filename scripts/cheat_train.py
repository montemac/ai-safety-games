# %%
# Imports, etc
from typing import Dict, List, Any
import pickle

import numpy as np
import pandas as pd
import torch as t
import plotly.express as px
from tqdm.auto import tqdm

from ai_safety_games import cheat_utils, utils

utils.enable_ipython_reload()

_ = t.set_grad_enabled(True)


# %%
# Load game data and train models!

# Setup constants
# DATASET_FOLDER = "../datasets/random_dataset_20230731T001342"
DATASET_FOLDER = "../datasets/random_dataset_20230815T235622"
DEVICE = "cuda:0"
SEQUENCE_MODE = "tokens_score"

INCLUDED_PLAYERS = [0, 3]
CHEATING_PLAYERS = [3]
# INCLUDE_CHEAT_RATES = [0.001, 0.002, 0.004, 0.007]
INCLUDE_CHEAT_RATES = [1.0]

# CHEAT_PENALTY_WEIGHTS = [0.1, 0.3, 1.0, 3.0, 10.0, 30, 100]
# CHEAT_PENALTY_APPLY_PROBS = [0.01, 0.03, 0.1, 0.3, 1.0]
CHEAT_PENALTY_WEIGHTS = [0.0]
CHEAT_PENALTY_APPLY_PROBS = [0.0]

results_list = []
for include_cheat_rate in tqdm(INCLUDE_CHEAT_RATES):
    # TEMP: comment out for now
    # def game_filter(summary_lists: Dict[str, List[Any]]) -> List[int]:
    #     """Filter out games that don't match criteria"""
    #     inds = []
    #     for idx, player_inds in enumerate(summary_lists["player_indices"]):
    #         if all(
    #             [player_ind in INCLUDED_PLAYERS for player_ind in player_inds]
    #         ):
    #             if any(
    #                 player_ind in CHEATING_PLAYERS
    #                 for player_ind in player_inds
    #             ):
    #                 if np.random.rand() < include_cheat_rate:
    #                     inds.append(idx)
    #             else:
    #                 inds.append(idx)
    #     return inds

    # game_data = cheat_utils.load_game_data(
    #     dataset_folder=DATASET_FOLDER,
    #     sequence_mode=SEQUENCE_MODE,
    #     game_filter=game_filter,
    #     device=DEVICE,
    # )

    # Quick dataset analysis
    # loaded_summary = game_data.summary.iloc[game_data.loaded_game_inds]
    # player_tuples = loaded_summary["player_indices"].apply(lambda x: tuple(x))
    # first_player = player_tuples.apply(lambda x: x[0]).rename("first_player")

    # def scores_to_win_pos(scores):
    #     """Convert scores to winning player position"""
    #     scores = [s for idx, s in scores]
    #     arg_sort = np.argsort(scores)
    #     if scores[arg_sort[-1]] == scores[arg_sort[-2]]:
    #         return -1
    #     else:
    #         return arg_sort[-1]

    # winning_pos = (
    #     loaded_summary["scores"].apply(scores_to_win_pos).rename("winning_pos")
    # )

    # win_rate_by_first_player = (winning_pos == 0).groupby([first_player]).mean()

    # Train using new high-level function
    for cheat_penalty_weight in tqdm(CHEAT_PENALTY_WEIGHTS):
        for cheat_penalty_apply_prob in tqdm(CHEAT_PENALTY_APPLY_PROBS):
            results, game_data, test_func = cheat_utils.train(
                cheat_utils.CheatTrainingConfig(
                    dataset_folder=DATASET_FOLDER,
                    sequence_mode=SEQUENCE_MODE,
                    game_filter=game_filter,
                    device=DEVICE,
                    cached_game_data=game_data,
                    train_fraction=0.99,
                    n_layers=1,
                    d_model=16,
                    d_head=8,
                    attn_only=True,
                    n_ctx=199,  # TODO: don't make this a constant!
                    epochs=200,
                    # epochs=int(10 * 125000 / len(game_data.loaded_game_inds)),
                    batch_size=1000,
                    lr=0.001,
                    # lr_schedule=("cosine_with_warmup", {"warmup_fraction": 0.05}),
                    lr_schedule=None,
                    weight_decay=0,
                    log_period=500000,
                    seed=1,
                    test_player_inds=INCLUDED_PLAYERS,
                    test_goal_scores=[0, 5],
                    cheat_penalty_weight=cheat_penalty_weight,
                    cheat_penalty_apply_prob=cheat_penalty_apply_prob,
                    cheat_penalty_min_prob=0.1,
                )
            )
            results_list.append(
                {
                    "include_cheat_rate": include_cheat_rate,
                    "cheat_penalty_weight": cheat_penalty_weight,
                    "cheat_penalty_apply_prob": cheat_penalty_apply_prob,
                    "num_games": len(game_data.loaded_game_inds),
                    "results": results,
                }
            )


# %%
# Show training results
px.line(
    results.results,
    x="batch",
    y=[
        "loss_train",
        "loss_test",
        # "test_margin_mean_goal_2",
        # "test_margin_mean_goal_5",
        "test_win_rate_goal_0",
        "test_win_rate_goal_5",
    ],
    title="Training loss",
).show()

# %%
# Run some more tests
test_players = [game_data.players[idx] for idx in INCLUDED_PLAYERS]

goal_score = 5
margins_list = []
for player in [results.model] + test_players:
    # for goal_score in [5]:
    margins = cheat_utils.run_test_games(
        model=player,
        game_config=game_data.game_config,
        num_games=500,
        goal_score=goal_score,
        max_turns=max(game_data.summary["turn_cnt"]),
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


# %%
# TEMP: compare a few specific training runs
RUNS = [
    "20230829T170600",
    "20230829T185516",
    "20230829T205334",
    "20230830T002809",
    "20230830T020437",
    "20230830T034643",
    "20230830T052703",
    "20230830T132501",
    "20230830T151902",
    "20230830T170157",
    "20230830T185358",
    "20230831T023701",
    "20230831T072206",
    "20230831T113929",
    "20230907T160608",
]

game_filter = None

results_list = []
model_descs = []
for run in RUNS:
    with open(f"cheat_train_results/{run}/results.pkl", "rb") as file:
        results = pickle.load(file)
    model = results["training_results"]["model"]
    model_descs.append(
        f"{model.cfg.n_layers}L, "
        f"D:{model.cfg.d_model}, "
        f"H:{model.cfg.d_head}, "
        f"C:{model.cfg.n_ctx}, "
        f"A:{model.cfg.attn_only}"
        f"<br>{run}"
    )
    results_list.append(results["training_results"]["results"])

results_all = pd.concat(
    results_list, axis=0, keys=model_descs, names=["model_desc"]
).reset_index()

for value_vars, title in [
    (["loss_train", "loss_test"], "Loss"),
    (["test_win_rate_goal_0", "test_win_rate_goal_5"], "Win rate"),
]:
    plot_df = results_all.melt(
        id_vars=["model_desc", "batch"],
        value_vars=value_vars,
        var_name="indicator",
        value_name="value",
    )
    px.line(
        plot_df,
        x="batch",
        y="value",
        color="model_desc",
        facet_col="indicator",
        facet_col_wrap=2,
        title=title,
    ).show()

# %%
# Imports, etc.

# import pickle
# import datetime
# import glob
# import os
# import lzma

# # TEMP
# # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# from sortedcontainers import SortedList
# from collections import defaultdict
# import numpy as np
# import pandas as pd
# import torch as t
# import torch.optim as optim
# from torch.utils.data import random_split
# from tqdm.auto import tqdm
# import plotly.express as px
# from transformer_lens import HookedTransformerConfig

# from ai_safety_games import cheat, utils, training
# from ai_safety_games.ScoreTransformer import (
#     ScoreTransformer,
#     ScoreTransformerConfig,
# )

# utils.enable_ipython_reload()

# _ = t.set_grad_enabled(True)

# # %%
# # Load a dataset into memory
# FOLDER = "../datasets/random_dataset_20230731T001342"
# DEVICE = "cuda:0"

# # Load config info
# with open(os.path.join(FOLDER, "config.pkl"), "rb") as file:
#     config_dict = pickle.load(file)
#     game_config = cheat.CheatConfig(**config_dict["game.config"])
#     game = cheat.CheatGame(game_config)

# # Load summary info
# with open(os.path.join(FOLDER, "summary.pkl"), "rb") as file:
#     summary_lists = pickle.load(file)
#     # TODO: explain where all these numbers come from :)
#     # max_seq_len = (
#     #     (max(summary_lists["turn_cnt"]) + 1)
#     #     // game_config.num_players
#     #     * (2 * game.config.num_players + game.config.num_ranks)
#     # ) + 2


# # Get the token vocab
# vocab, player_action_vocab = game.get_token_vocab()
# vocab_strs = {idx: tok for tok, idx in vocab.items()}

# # Load games and convert to token tensors plus scores
# tokens_list = []
# scores_list = []
# for game_idx, filename in enumerate(
#     tqdm(sorted(glob.glob(os.path.join(FOLDER, "game_*.pkl")))[:200000])
# ):
#     with lzma.open(filename, "rb") as file:
#         game_results = pickle.load(file)
#         # Rebuild state history
#         state_history = [
#             cheat.CheatState(**state_dict)
#             for state_dict in game_results["state_history_list"]
#         ]
#         # Calculate scores from state history (there was a bug in scores
#         # calculation before so don't use loaded scores)
#         hand_sizes = np.array([len(hand) for hand in state_history[-1].hands])
#         scores = -hand_sizes
#         next_best_score = np.max(scores[scores != 0])
#         scores[scores == 0] = -next_best_score
#         # Get token sequences
#         tokens_this, _, _ = cheat.get_seqs_from_state_history(
#             game=game, vocab=vocab, state_history=state_history
#         )
#         tokens_list.extend([row for row in tokens_this])
#         scores_list.append(scores)

# # Convert to single tokens tensor, scores tensor, and seq lengths tensor
# # Use pad_sequence to pad to the max length of any game
# tokens_all = t.nn.utils.rnn.pad_sequence(
#     tokens_list, batch_first=True, padding_value=0
# ).to(DEVICE)
# scores_all = t.tensor(np.array(scores_list).flatten(), dtype=t.float32).to(
#     DEVICE
# )
# seq_lens_all = t.tensor([len(toks) for toks in tokens_list], dtype=t.int64).to(
#     DEVICE
# )

# # Positions corresponding to actions
# first_action_pos = (
#     (game_config.num_players - 1) * 2 + 2 + game.config.num_ranks
# )
# action_pos_step = game_config.num_players * 2 + game.config.num_ranks

# # Turn sequence lengths into a loss mask
# loss_mask = t.zeros_like(tokens_all, dtype=t.float32)
# for idx, seq_len in enumerate(seq_lens_all):
#     loss_mask[idx, :seq_len] = 1
# loss_mask = loss_mask[:, first_action_pos::action_pos_step]


# # %%


# # %%
# # TEMP: hack to fix rtgs to use victory_margin
# # for game_idx, filename in enumerate(
# #     tqdm(sorted(glob.glob(os.path.join(FOLDER, "rsas_*.pkl"))))
# # ):
# #     with open(filename, "rb") as file:
# #         reward, state, action = pickle.load(file)
# #     if t.max(reward).item() == 0:
# #         print("here")
# #         print(reward[0, 0].item())
# #         winning_player_idx = t.argmax(reward[:, 0]).item()
# #         other_indices = [
# #             idx for idx in range(reward.shape[0]) if idx != winning_player_idx
# #         ]
# #         best_other_score = t.max(reward[other_indices, 0]).item()
# #         reward[winning_player_idx, :] = -best_other_score
# #         # Save the modified reward tensor and other tensors back to disk
# #         with open(filename, "wb") as file:
# #             pickle.dump((reward, state, action), file)


# # %%
# # Training loop using library function
# TRAINING_MINS = 20
# BATCH_SIZE = 100
# LOG_PERIOD = 10000
# N_LAYERS = 8
# D_MODEL = 128
# D_HEAD = 16
# ATTN_ONLY = False
# LR = 0.001
# WEIGHT_DECAY = 0.00
# SEED = 0

# # Split data into train and test sets
# # TODO: probably a simpler way to do this
# generator = t.Generator().manual_seed(SEED)
# train_inds, test_inds = [
#     t.tensor(subset)
#     for subset in random_split(
#         range(tokens_all.shape[0]), [0.8, 0.2], generator=generator
#     )
# ]

# # Initialize a simple test model
# model = ScoreTransformer(
#     cfg=HookedTransformerConfig(
#         n_layers=N_LAYERS,
#         d_model=D_MODEL,
#         d_head=D_HEAD,
#         d_vocab=len(vocab),
#         d_vocab_out=len(player_action_vocab),
#         act_fn="relu",
#         device=DEVICE,
#         seed=SEED,
#         n_ctx=seq_lens_all.max().item(),
#         attn_only=ATTN_ONLY,
#     ),
#     st_cfg=ScoreTransformerConfig(
#         first_action_pos=first_action_pos, action_pos_step=action_pos_step
#     ),
# )

# # Standard test function
# test_inds_small = test_inds[:1000]
# test_func = training.make_standard_test_func(
#     test_data=training.TrainingTensors(
#         inputs=[tokens_all[test_inds_small], scores_all[test_inds_small]],
#         output=tokens_all[test_inds_small, first_action_pos::action_pos_step]
#         .detach()
#         .clone(),
#         loss_mask=loss_mask[test_inds_small],
#     ),
#     test_batch_size=BATCH_SIZE,
# )

# # Train!
# results = training.train_custom_transformer(
#     model=model,
#     config=training.TrainingConfig(
#         project_name="cheat",
#         training_mins=TRAINING_MINS,
#         batch_size=BATCH_SIZE,
#         lr=LR,
#         weight_decay=WEIGHT_DECAY,
#         log_period=LOG_PERIOD,
#         seed=SEED,
#         save_results=True,
#     ),
#     training_data=training.TrainingTensors(
#         inputs=[tokens_all[train_inds], scores_all[train_inds]],
#         output=tokens_all[train_inds, first_action_pos::action_pos_step]
#         .detach()
#         .clone(),
#         loss_mask=loss_mask[train_inds],
#     ),
#     test_func=test_func,
# )

# # %%
# # Show training results
# plot_df = results.results.melt(
#     id_vars=["elapsed_time"],
#     value_vars=["loss_train", "loss_test"],
#     var_name="loss_type",
#     value_name="loss",
# )
# px.line(
#     plot_df,
#     x="elapsed_time",
#     y="loss",
#     color="loss_type",
#     title="Training loss",
# ).show()


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

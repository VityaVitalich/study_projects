import math
import os
import shutil
import string
import argparse
import time
from collections import defaultdict
from typing import List, Tuple, TypeVar, Optional, Callable, Iterable

import arpa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import wandb
from matplotlib.colors import LogNorm
from torch import optim
from tqdm.notebook import tqdm
import utils as utils

from train_utils import Collate, train, test

from models import CTCDNN, CTCRNN, CTCBiRNN, DeepSpeech

from alignment import tokenizer, BLANK_SYMBOL
import yaml


import warnings

warnings.filterwarnings("ignore")

# folder to load config file
CONFIG_PATH = "./config/"


parser = argparse.ArgumentParser(description="Training ASR")
parser.add_argument("config_name", type=str, help="name of config file")
args = parser.parse_args()

name = args.config_name


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config(name)


train_dataset = torchaudio.datasets.LIBRISPEECH(
    "./data", url="train-clean-100", download=True
)
test_dataset = torchaudio.datasets.LIBRISPEECH(
    "./data", url="test-clean", download=True
)

#!L
# pragma async
# PRAGMA ASYNC IS NECESSARY FOR TRAINING!
torch.manual_seed(7)
if torch.cuda.is_available():
    print("GPU found! 🎉")
    device = "cuda:0"
else:
    print("Only CPU found! 💻")
    device = "cpu"

verbose = False

# Hyperparameters for your model
hparams = {
    "n_cnn_layers": config["cnn_n_layers"],
    "n_rnn_layers": config["rnn_n_layers"],
    "rnn_dim": config["rnn_dim"],
    "n_class": 29,
    "n_feats": config["n_feats"],
    "stride": config["stride"],
    "dropout": config["dropout"],
    "learning_rate": 3e-4,
    "batch_size": config["bs"],
    "epochs": config["epochs"],
}

train_collate_fn = Collate(data_type="train")
test_collate_fn = Collate(data_type="test")

# Define Dataloyour training and test data loaders
kwargs = {"num_workers": 4, "pin_memory": True} if device == "cuda" else {}
train_loader = data.DataLoader(
    train_dataset,
    batch_size=hparams["batch_size"],
    shuffle=True,
    collate_fn=train_collate_fn,
    **kwargs
)

kwargs = {"num_workers": 1, "pin_memory": True} if device == "cuda" else {}
test_loader = data.DataLoader(
    test_dataset,
    batch_size=hparams["batch_size"],
    shuffle=False,
    collate_fn=test_collate_fn,
    **kwargs
)

wandb.init(project="hw2-dlaudio", group=config["model_type"], config=hparams)


# Train a non-recurrent model
model = eval(config["model_type"])(
    hparams["n_cnn_layers"],
    hparams["n_rnn_layers"],
    hparams["rnn_dim"],
    hparams["n_class"],
    hparams["n_feats"],
    hparams["stride"],
    hparams["dropout"],
)
# ctc_dnn.to(device)
if config["parallel"]:
    model = torch.nn.parallel.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()
else:
    model = model.cuda()


optimizer = torch.optim.AdamW(
    model.parameters(), lr=hparams["learning_rate"]
)  # YOUR CODE  - SUGGESTED ADAM/ADAMW
criterion = nn.CTCLoss(blank=tokenizer.get_symbol_index(BLANK_SYMBOL), reduction="mean")
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4, epochs=hparams["epochs"], steps_per_epoch=len(train_loader)
)  # YOUR CODE  - SUGGESTED ONE CYCLE


for epoch in tqdm(range(1, hparams["epochs"] + 1)):
    train(model, device, train_loader, criterion, optimizer, scheduler, epoch)

    saving_name = config["name"] + str(epoch) + ".tar"
    utils.save_checkpoint(model, checkpoint_name=saving_name)
    wandb.save(saving_name)
    if epoch % 10 == 0:
        test(model, device, test_loader, criterion, epoch, "Greedy", "dnn")


utils.save_checkpoint(model, checkpoint_name=config["name"] + ".tar")

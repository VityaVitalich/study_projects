import math
import os
import shutil
import string
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

from alignment import greedy_decoder, beam_search_decoder, tokenizer, LanguageModel


train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(),
    # Can add data augmentation here!
    torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
    torchaudio.transforms.TimeMasking(time_mask_param=30),
)

test_audio_transforms = nn.Sequential(
    # Add feature transformations here
    torchaudio.transforms.MelSpectrogram()
)


class Collate:
    def __init__(self, data_type="test") -> None:
        super(Collate, self).__init__()

        self.data_type = data_type

    def __call__(
        self, data: torchaudio.datasets.librispeech.LIBRISPEECH
    ) -> Tuple[List[torch.Tensor], ...]:
        """
        :param data: is a list of tuples of [features, label], where features has dimensions [n_features, length]
        "returns features, lengths, labels:
              features is a Tensor [batchsize, features, max_length]
              lengths is a Tensor of lengths [batchsize]
              labels is a Tesnor of targets [batchsize]
        """

        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []
        for waveform, _, utterance, _, _, _ in data:
            if self.data_type == "train":
                spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            elif self.data_type == "test":
                spec = test_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            else:
                raise Exception("data_type should be train or valid")
            spectrograms.append(spec)
            label = torch.Tensor(tokenizer.text_to_indices(utterance.lower()))
            labels.append(label)
            input_lengths.append(spec.shape[0] // 2)
            label_lengths.append(len(label))

        spectrograms = (
            nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
            .unsqueeze(1)
            .transpose(2, 3)
        )
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return spectrograms, labels, input_lengths, label_lengths


def train(
    model: nn.Module,
    device: str,
    train_loader: data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    epoch: int,
) -> None:
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.cuda(), labels.cuda()

        optimizer.zero_grad()

        output = model(spectrograms, input_lengths)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        scheduler.step()
        if batch_idx % 50 == 0 or batch_idx == data_len:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(spectrograms),
                    data_len,
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            wandb.log({"loss_train": loss.item()})

        # del spectrograms
        # del labels
        # torch.cuda.empty_cache()


def test(
    model: nn.Module,
    device: str,
    test_loader: data.DataLoader,
    criterion: nn.Module,
    epoch: int,
    decode: str = "Greedy",
    lm: LanguageModel = None,
    save_path: str = None,
) -> None:
    print("Beginning eval...")
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        start = time.time()
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            matrix = model(spectrograms, input_lengths)  # (batch, time, n_class)
            matrix = F.log_softmax(matrix, dim=2)
            probs = F.softmax(matrix, dim=2)
            matrix = matrix.transpose(0, 1)  # (time, batch, n_class)

            if i == 3:
                np.savetxt(f"{save_path}_matrix.txt", probs[0].cpu().numpy())
                np.savetxt(f"{save_path}_labels.txt", labels[0].cpu().numpy())

            loss = criterion(matrix, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            # print(decode)
            if decode == "Greedy":
                decoded_preds, decoded_targets = greedy_decoder(
                    matrix.transpose(0, 1), labels, label_lengths
                )
                # print(len(decoded_preds))
            elif decode == "BeamSearch":
                ## THIS IS THE FUNCTION YOU SHOULD IMPLEMENT
                decoded_preds, decoded_targets = beam_search_decoder(
                    probs, labels, label_lengths, input_lengths, lm=lm
                )
            for j in range(len(decoded_preds)):
                test_cer.append(utils.cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(utils.wer(decoded_targets[j], decoded_preds[j]))

            # del spectrograms
            # del labels
            # torch.cuda.empty_cache()

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)
    wandb.log({"loss_test": test_loss, "avg_cer": avg_cer, "avg_wer": avg_wer})
    print(
        "Epoch: {:d}, Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n".format(
            epoch, test_loss, avg_cer, avg_wer
        )
    )

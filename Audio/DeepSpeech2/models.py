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
import yaml


######### DEEP SPEECH EXTRACTOR #################


class DSFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_cnn_layers: int,
        n_rnn_layers: int,
        rnn_dim: int,
        n_feats: int,
        stride: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super(DSFeatureExtractor, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        )
        self.fully_connected = nn.Linear(1024, rnn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        return x


######### DNN #################


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearBlock, self).__init__()

        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class CTCDNN(nn.Module):
    def __init__(
        self,
        n_cnn_layers: int,
        n_rnn_layers: int,
        rnn_dim: int,
        n_class: int,
        n_feats: int,
        stride: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super(CTCDNN, self).__init__()

        self.feature_extractor = DSFeatureExtractor(
            n_cnn_layers, n_rnn_layers, rnn_dim, n_feats, stride, dropout
        )

        self.intermediate_layers = nn.Sequential(
            *[LinearBlock(rnn_dim, rnn_dim) for i in range(n_rnn_layers)]
        )

        self.classifier = nn.Linear(rnn_dim, n_class)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.intermediate_layers(x)
        x = self.classifier(x)
        return x


####### RNN #############


class CTCRNN(nn.Module):
    def __init__(
        self,
        n_cnn_layers: int,
        n_rnn_layers: int,
        rnn_dim: int,
        n_class: int,
        n_feats: int,
        stride: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super(CTCRNN, self).__init__()

        self.feature_extractor = DSFeatureExtractor(
            n_cnn_layers, n_rnn_layers, rnn_dim, n_feats, stride, dropout
        )

        self.intermediate_layers = nn.Sequential(
            *[
                nn.LSTM(rnn_dim, rnn_dim, bidirectional=False, bias=True)
                for i in range(n_rnn_layers)
            ]
        )

        self.classifier = nn.Linear(rnn_dim, n_class)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        for rnn in self.intermediate_layers:
            x, _ = rnn(x)
        x = self.classifier(x)
        return x


class DeepSpeech(nn.Module):
    def __init__(
        self,
        n_cnn_layers: int,
        n_rnn_layers: int,
        rnn_dim: int,
        n_class: int,
        n_feats: int,
        stride: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super(DeepSpeech, self).__init__()

        self.feature_extractor = DSFeatureExtractor(
            n_cnn_layers, n_rnn_layers, rnn_dim, n_feats, stride, dropout
        )

        self.intermediate_layers = nn.Sequential(
            nn.LSTM(rnn_dim, rnn_dim, bidirectional=True, bias=True),
            *[
                nn.LSTM(rnn_dim * 2, rnn_dim, bidirectional=True, bias=True)
                for i in range(n_rnn_layers - 1)
            ]
        )

        self.classifier = nn.Linear(2 * rnn_dim, n_class)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        for rnn in self.intermediate_layers:
            x, _ = rnn(x)
        x = self.classifier(x)
        return x


#################### FANCY CNN RNN LIKE ########################
class CNNLayerNorm(nn.Module):
    """Layer normalization built for CNNs input"""

    def __init__(self, n_feats: int) -> None:
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
    except with layer norm instead of batch norm
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        stride: int,
        dropout: float,
        n_feats: int,
    ) -> None:
        super(ResidualCNN, self).__init__()

        self.net = nn.Sequential(
            CNNLayerNorm(n_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
            ),
            CNNLayerNorm(n_feats),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x  # (batch, channel, feature, time)
        x = self.net(x)
        x += residual
        return x  # (batch, channel, feature, time)


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        n_cnn_layers: int,
        n_rnn_layers: int,
        rnn_dim: int,
        n_feats: int,
        stride: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super(FeatureExtractor, self).__init__()

        n_feats = n_feats // 2
        self.cnn = nn.Conv2d(
            1, 32, 3, stride=stride, padding=3 // 2
        )  # cnn for extracting heirachal features

        self.rescnn_layers = nn.Sequential(
            *[
                ResidualCNN(
                    in_channels=32,
                    out_channels=32,
                    kernel=3,
                    stride=1,
                    dropout=dropout,
                    n_feats=n_feats,
                )
                for _ in range(n_cnn_layers)
            ]
        )
        self.fully_connected = nn.Linear(n_feats * 32, rnn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        # print(x.size())
        x = self.fully_connected(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearBlock, self).__init__()

        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class CTCDNN(nn.Module):
    def __init__(
        self,
        n_cnn_layers: int,
        n_rnn_layers: int,
        rnn_dim: int,
        n_class: int,
        n_feats: int,
        stride: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super(CTCDNN, self).__init__()

        self.feature_extractor = FeatureExtractor(
            n_cnn_layers, n_rnn_layers, rnn_dim, n_feats, stride, dropout
        )

        self.intermediate_layers = nn.Sequential(
            *[LinearBlock(rnn_dim, rnn_dim) for i in range(n_rnn_layers)]
        )

        self.classifier = nn.Linear(rnn_dim, n_class)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.intermediate_layers(x)
        x = self.classifier(x)
        return x


class RNNBlock(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(RNNBlock, self).__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(rnn_dim),
            nn.GELU(),
            nn.GRU(
                input_size=rnn_dim,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=batch_first,
                bidirectional=True,
            ),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.net(x)
        x = self.dropout(x)
        return x


class CTCBiRNN(nn.Module):
    def __init__(
        self,
        n_cnn_layers: int,
        n_rnn_layers: int,
        rnn_dim: int,
        n_class: int,
        n_feats: int,
        stride: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super(CTCBiRNN, self).__init__()

        self.feature_extractor = FeatureExtractor(
            n_cnn_layers, n_rnn_layers, rnn_dim, n_feats, stride, dropout
        )

        self.intermediate_layers = nn.Sequential(
            RNNBlock(
                rnn_dim=rnn_dim, hidden_size=rnn_dim, dropout=dropout, batch_first=True
            ),
            *[
                RNNBlock(
                    rnn_dim=rnn_dim * 2,
                    hidden_size=rnn_dim,
                    dropout=dropout,
                    batch_first=False,
                )
                for i in range(n_rnn_layers - 1)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class),
        )

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.intermediate_layers(x)
        x = self.classifier(x)
        return x

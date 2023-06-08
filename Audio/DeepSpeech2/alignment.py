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


class Tokenizer:
    """
    Maps characters to integers and vice versa
    """

    def __init__(self):
        self.char_map = {}
        self.index_map = {}
        for i, ch in enumerate(
            ["'", " "] + list(string.ascii_lowercase) + [BLANK_SYMBOL]
        ):
            self.char_map[ch] = i
            self.index_map[i] = ch

    def text_to_indices(self, text: str) -> List[int]:
        return [self.char_map[ch] for ch in text]

    def indices_to_text(self, labels: List[int]) -> str:
        return "".join([self.index_map[i] for i in labels])

    def get_symbol_index(self, sym: str) -> int:
        return self.char_map[sym]


BLANK_SYMBOL = "_"
tokenizer = Tokenizer()
NEG_INF = -float("inf")
LanguageModel = TypeVar("LanguageModel")


def logsumexp(*args) -> float:
    """
    Log-sum-exp trick for log-domain calculations
    See for details: https://en.wikipedia.org/wiki/LogSumExp
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def modify_sequence(sequence: List[int], blank_idx: int) -> List[int]:
    """
    Modifies sequence which with START, END blanks and between each character
    """
    modified_sequence = []

    for idx in sequence:
        modified_sequence += [blank_idx, idx]

    modified_sequence.append(blank_idx)
    return modified_sequence


def forward_algorithm(sequence: List[int], matrix: np.ndarray) -> np.ndarray:
    """
    :param sequence: a string converted to an index array by Tokenizer
    :param matrix: A matrix of shape (K, T) with probability distributions over phonemes at each moment of time.
    :return: the result of the forward pass of shape (2 * len(sequence) + 1, T)
    """
    # Turn probs into log-probs
    matrix = np.log(matrix)

    blank = tokenizer.get_symbol_index(BLANK_SYMBOL)
    mod_sequence = modify_sequence(sequence, blank)

    # Initialze
    # (2L + 1) x T
    alphas = np.full([len(mod_sequence), matrix.shape[1]], NEG_INF)

    for t in range(matrix.shape[1]):
        for s in range(len(mod_sequence)):
            # First Step
            ch = mod_sequence[s]
            if t == 0:
                if s != 0 and s != 1:
                    alphas[s][t] = NEG_INF
                else:
                    alphas[s][t] = matrix[ch][t]

            # Upper diagonal zeros
            elif s < alphas.shape[0] - 2 * (alphas.shape[1] - t) - 1:  # CONDITION
                alphas[s][t] = NEG_INF
            else:
                # Need to do this stabily
                if s == 0:
                    alphas[s][t] = alphas[s][t - 1] + matrix[ch][t]
                elif s == 1:
                    alphas[s][t] = (
                        logsumexp(alphas[s][t - 1], alphas[s - 1][t - 1])
                        + matrix[ch][t]
                    )
                else:
                    if ch == blank or ch == mod_sequence[s - 2]:
                        alphas[s][t] = (
                            logsumexp(alphas[s][t - 1], alphas[s - 1][t - 1])
                            + matrix[ch][t]
                        )
                    else:
                        alphas[s][t] = (
                            logsumexp(
                                alphas[s][t - 1],
                                alphas[s - 1][t - 1],
                                alphas[s - 2][t - 1],
                            )
                            + matrix[ch][t]
                        )
    return alphas


def backward_algorithm(sequence: List[int], matrix: np.ndarray) -> np.ndarray:
    """
    :param sequence: a string converted to an index array by Tokenizer
    :param matrix: A matrix of shape (K, T) with probability distributions over phonemes at each moment of time.
    :return: the result of the backward pass of shape (2 * len(sequence) + 1, T)
    """
    matrix = np.log(matrix)
    blank = tokenizer.get_symbol_index(BLANK_SYMBOL)
    mod_sequence = modify_sequence(sequence, blank)
    betas = np.full([len(mod_sequence), matrix.shape[1]], NEG_INF)

    for t in reversed(range(matrix.shape[1])):
        for s in reversed(range(len(mod_sequence))):
            # First Step
            ch = mod_sequence[s]
            if t == matrix.shape[1] - 1:
                if s == betas.shape[0] - 1 or s == betas.shape[0] - 2:
                    betas[s][t] = 0

            # Lower Diagonal Zeros
            elif s > 2 * t + 1:  # CONDITION
                betas[s][t] = NEG_INF
            else:
                if s == len(mod_sequence) - 1:
                    betas[s][t] = betas[s][t + 1] + matrix[ch][t]
                elif s == len(mod_sequence) - 2:
                    betas[s][t] = (
                        logsumexp(betas[s][t + 1], betas[s + 1][t + 1]) + matrix[ch][t]
                    )
                else:
                    if ch == blank or ch == mod_sequence[s + 2]:
                        betas[s][t] = (
                            logsumexp(betas[s][t + 1], betas[s + 1][t + 1])
                            + matrix[ch][t]
                        )
                    else:
                        betas[s][t] = (
                            logsumexp(
                                betas[s][t + 1],
                                betas[s + 1][t + 1],
                                betas[s + 2][t + 1],
                            )
                            + matrix[ch][t]
                        )
    return betas


def soft_alignment(labels_indices: List[int], matrix: np.ndarray) -> np.ndarray:
    """
    Returns the alignment coefficients for the input sequence
    """
    alphas = forward_algorithm(labels_indices, matrix)
    betas = backward_algorithm(labels_indices, matrix)

    # Move from log space back to prob space
    align = np.exp(alphas + betas)

    # Normalize Alignment
    align = align / np.sum(align, axis=0)

    return align


#!L
def greedy_decoder(
    output: torch.Tensor,
    labels: List[torch.Tensor],
    label_lengths: List[int],
    collapse_repeated: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param output: torch.Tensor of Probs or Log-Probs of shape [batch, time, classes]
    :param labels: list of label indices converted to torch.Tensors
    :param label_lengths: list of label lengths (without padding)
    :param collapse_repeated: whether the repeated characters should be deduplicated
    :return: the result of the decoding and the target sequence
    """
    blank_label = tokenizer.get_symbol_index(BLANK_SYMBOL)

    # Get max classes
    ########################
    arg_maxes = output.argmax(dim=-1)
    ########################

    decodes = []
    targets = []

    # For targets and decodes remove repeats and blanks
    for i, args in enumerate(arg_maxes):
        decode = []
        true_labels = labels[i][: label_lengths[i]].tolist()
        targets.append(tokenizer.indices_to_text(true_labels))

        # Remove repeats, then remove blanks
        for j, index in enumerate(args):
            ########################
            if j != 0:
                if index == args[j - 1]:
                    continue
            decode.append(int(index.cpu().detach()))
            ########################
        ####
        decode = [x for x in decode if x != blank_label]
        ######

        decodes.append(tokenizer.indices_to_text(decode))
    return decodes, targets


class Beam:
    def __init__(self, beam_size: int) -> None:
        self.beam_size = beam_size

        fn = lambda: (NEG_INF, NEG_INF)

        # Store probs key - prefix, value - p_blank, p_not_blank for ? step
        self.candidates = defaultdict(fn)

        # Store sorted by cumulative probability self.candidates
        self.top_candidates_list = [
            (tuple(), (0.0, NEG_INF))  # log(p_blank) = 0, log(p_not_blank) = -inf
        ]

    def get_probs_for_prefix(self, prefix: Tuple[int]) -> Tuple[float, float]:
        p_blank, p_not_blank = self.candidates[prefix]
        return p_blank, p_not_blank

    def update_probs_for_prefix(
        self, prefix: Tuple[int], next_p_blank: float, next_p_not_blank: float
    ) -> None:
        self.candidates[prefix] = (next_p_blank, next_p_not_blank)

    def update_top_candidates_list(self) -> None:
        top_candidates = sorted(
            self.candidates.items(), key=lambda x: logsumexp(*x[1]), reverse=True
        )
        self.top_candidates_list = top_candidates[: self.beam_size]


def calculate_probability_score_with_lm(lm: LanguageModel, prefix: str) -> float:
    text = (
        tokenizer.indices_to_text(prefix).upper().strip()
    )  # Use upper case for LM and remove the trailing space
    lm_prob = lm.log_p(text)
    score = lm_prob / np.log10(np.e)  # Convert to natural log, as ARPA LM uses log10
    return score


#!L


def decode(
    probs: np.ndarray,
    beam_size: int = 5,
    lm: Optional[LanguageModel] = None,
    prune: float = 1e-5,
    alpha: float = 0.1,
    beta: float = 2,
):
    """
    :param probs: A matrix of shape (T, K) with probability distributions over phonemes at each moment of time.
    :param beam_size: the size of beams
    :lm: arpa language model
    :prune: the minimal probability for a symbol at which it can be added to a prefix
    :alpha: the parameter to de-weight the LM probability
    :beta: the parameter to up-weight the length correction term
    :return: the prefix with the highest sum of probabilites P_blank and P_not_blank
    """
    T, S = probs.shape
    probs = np.log(probs)
    blank = tokenizer.get_symbol_index(BLANK_SYMBOL)
    space = tokenizer.get_symbol_index(" ")
    prune = NEG_INF if prune == 0.0 else np.log(prune)

    beam = Beam(beam_size)
    # Итерируемся по оси времени
    for t in range(T):
        next_beam = Beam(beam_size)

        # Итерируемся по символам
        for s in range(S):
            p = probs[t, s]
            # Prune the vocab - пропускаем символ, если вероятность оказаться в нем слишком мала на t-м щаге
            if p < prune:
                continue

            # Итерируемся по варинатам, в которые можем пойти из текущего символа
            # Сначала идут наиболее вероятные по сумме log(p_blank + p_not_blank) префиксы
            # (p_blank, p_not_blank) - вероятности на предыдущем t-1 шаге
            for prefix, (p_blank, p_not_blank) in beam.top_candidates_list:
                # Текущий символ - бланк
                if s == blank:
                    # вероятности на текущем шаге
                    p_b, p_nb = next_beam.get_probs_for_prefix(prefix)
                    next_beam.update_probs_for_prefix(
                        prefix=prefix,
                        next_p_blank=logsumexp(p_b, p_blank + p, p_not_blank + p),
                        next_p_not_blank=p_nb,
                    )
                    continue

                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)

                # Повторяющийся символ
                if s == end_t:
                    # Предыдущий символ - бланк
                    p_b, p_nb = next_beam.get_probs_for_prefix(n_prefix)
                    next_beam.update_probs_for_prefix(
                        prefix=n_prefix,
                        next_p_blank=p_b,
                        next_p_not_blank=logsumexp(p_nb, p + p_blank),
                    )
                    # Предудщий символ не бланк
                    p_b, p_nb = next_beam.get_probs_for_prefix(prefix)
                    next_beam.update_probs_for_prefix(
                        prefix=prefix,
                        next_p_blank=p_b,
                        next_p_not_blank=logsumexp(p_nb, p + p_not_blank),
                    )
                elif s == space and end_t is not None and lm is not None:
                    # Символ - пробел и не первый, нужно применить языковую модель
                    p_b, p_nb = next_beam.get_probs_for_prefix(n_prefix)
                    score = calculate_probability_score_with_lm(lm, n_prefix)
                    length = len(tokenizer.indices_to_text(prefix))

                    next_beam.update_probs_for_prefix(
                        prefix=n_prefix,
                        next_p_blank=p_b,
                        next_p_not_blank=logsumexp(
                            p_nb,
                            p_blank + p + score * alpha + np.log(length) * beta,
                            p_not_blank + p + score * alpha + np.log(length) * beta,
                        ),
                    )
                else:
                    p_b, p_nb = next_beam.get_probs_for_prefix(n_prefix)
                    next_beam.update_probs_for_prefix(
                        prefix=n_prefix,
                        next_p_blank=p_b,
                        next_p_not_blank=logsumexp(p_nb, p_blank + p, p_not_blank + p),
                    )

        next_beam.update_top_candidates_list()
        beam = next_beam

    best = beam.top_candidates_list[0]
    return best[0], -logsumexp(*best[1])


def beam_search_decoder(
    probs: np.ndarray,
    labels: List[List[int]],
    label_lengths: List[int],
    input_lengths: List[int],
    lm: LanguageModel,
    beam_size: int = 5,
    prune: float = 1e-3,
    alpha: float = 0.1,
    beta: float = 0.1,
):
    probs = probs.cpu().detach().numpy()
    decodes, targets = [], []

    for i, prob in enumerate(probs):
        targets.append(
            tokenizer.indices_to_text(labels[i][: label_lengths[i]].tolist())
        )
        int_seq, _ = decode(
            prob[: input_lengths[i]],
            lm=lm,
            beam_size=beam_size,
            prune=prune,
            alpha=alpha,
            beta=beta,
        )
        decodes.append(tokenizer.indices_to_text(int_seq))

    return decodes, targets

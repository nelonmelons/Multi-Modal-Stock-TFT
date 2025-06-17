import os
from dataModule import Data_Day_Hourly_StocksPrice

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from dataModule import *

from tqdm import tqdm

class Embedder(nn.Module):
    """
    Takes in features of a day (7 hours, 5 features each), and embeds them into a higher-dimensional space.

    Assume for a 2 - layer MLP, the input is of shape (batch_size, 35) and the output is of shape (batch_size, embed_dim).

    More efficiemly, we can view such that input is (batch_size x seq_len, 5) and output is (batch_size x seq_len, embed_dim).
    """
    def __init__(self, input_dim=35, embed_dim=128):
        super(Embedder, self).__init__()
        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass through the embedder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, seq_len, embed_dim).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class GRN(nn.Module):
    """
    Gated Residual Network (GRN) for time series data.

    This module applies a two-layer MLP with a gating mechanism and residual connection.
    Input shape: (B, T, input_dim) → Output shape: (B, T, output_dim) # we recommend input_dim == output_dim

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int, optional): Dimension of the output features. Defaults to None, which sets it to input_dim.
        dropout (float, optional): Dropout rate. Defaults to 0.1.

    The GRN applies a two-layer MLP with a gating mechanism and residual connection.
    The first layer transforms the input to a hidden dimension, and the second layer projects it to the output dimension.
    """
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim if output_dim is not None else input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)
        self.gate = nn.Linear(self.output_dim, self.output_dim)

        self.res_proj = (
            nn.Identity() if input_dim == self.output_dim
            else nn.Linear(input_dim, self.output_dim)
        )

        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the GRN.

        :param x: torch.Tensor
        :return: torch.Tensor
        """
        residual = self.res_proj(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        gate = torch.sigmoid(self.gate(x))
        x = x * gate

        x = x + residual
        x = self.layer_norm(x)
        return x


class EmbedderGRNWrapper(nn.Module):
    """
    Wrapper that applies Embedder and GRN sequentially.
    Input shape: (B, T, input_dim) → Output shape: (B, T, embed_dim)
    """
    def __init__(self, input_dim=35, embed_dim=128):
        super().__init__()
        self.embedder = Embedder(input_dim=input_dim, embed_dim=embed_dim)
        self.grn = GRN(input_dim=embed_dim, hidden_dim=embed_dim)

    def forward(self, x):
        x = self.embedder(x)
        x = self.grn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim=128, output_dim=35):
        super().__init__()
        self.out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        return self.out(x)  # (B, 1, output_dim)


MAX_SEQ_LEN = 64  # Default max sequence length for padding

class dataTensors(list):
    """
    A tuple-like class to hold tensors for train, validation, and test sets.
    Includes data preparation, tensorization, summarization, and masking methods.
    Supports object chaining.
    """
    def __init__(self, train_input, train_target, val_input, val_target, test_input, test_target):
        super().__init__([train_input, train_target, val_input, val_target, test_input, test_target])
        self.train_input = train_input
        self.train_target = train_target
        self.val_input = val_input
        self.val_target = val_target
        self.test_input = test_input
        self.test_target = test_target

    @staticmethod
    def prepare_sequences(
        dfs: List[pd.DataFrame],
        word_len: int = 7,
        features_per_hour: int = 5,
        stride: int = 1,
        max_seq_len: int = MAX_SEQ_LEN,
        tag: str = None
    ):
        word_dim = word_len * features_per_hour
        sequences = []

        for df in dfs:
            df_numeric = df.select_dtypes(include=[np.number]).dropna()
            if len(df_numeric) < word_len * 2:
                continue

            words = [
                df_numeric[i:i + word_len].values.reshape(-1)
                for i in range(0, len(df_numeric), word_len)
                if len(df_numeric[i:i + word_len]) == word_len
            ]
            words = [torch.tensor(w, dtype=torch.float32) for w in words]

            for i in range(1, len(words)):
                x_seq = words[max(0, i - max_seq_len):i]
                y_target = words[i]
                if tag:
                    sequences.append((torch.stack(x_seq), y_target.unsqueeze(0), tag))
                else:
                    sequences.append((torch.stack(x_seq), y_target.unsqueeze(0)))

        return sequences

    @staticmethod
    def tensorize_and_pad(pairs: List[Tuple[torch.Tensor, torch.Tensor]], max_seq_len: int):
        print(f"Padding sequences to max length {max_seq_len}...")
        xs, ys = zip(*pairs)
        padded_x = torch.stack([
            torch.cat([x, torch.zeros(max_seq_len - x.size(0), x.size(1))])
            if x.size(0) < max_seq_len else x[-max_seq_len:]
            for x in xs
        ])
        ys = torch.stack(ys)

        perm = torch.randperm(len(padded_x))
        return padded_x[perm], ys[perm]

    @classmethod
    def from_dir(cls, dir_path: str, word_len: int = 7, features_per_hour: int = 5, max_seq_len: int = MAX_SEQ_LEN):
        data = Data_Day_Hourly_StocksPrice.from_dir(dir_path)

        train_pairs = cls.prepare_sequences(data['train'], word_len=word_len, features_per_hour=features_per_hour, max_seq_len=max_seq_len, tag="train")
        val_pairs = cls.prepare_sequences(data['validation'], word_len=word_len, features_per_hour=features_per_hour, max_seq_len=max_seq_len, tag="val")

        combined = data['train'] + data['test']
        test_pairs_full = cls.prepare_sequences(combined, word_len=word_len, features_per_hour=features_per_hour, max_seq_len=max_seq_len)

        test_targets_set = set()
        for df in data['test']:
            df_numeric = df.select_dtypes(include=[np.number]).dropna()
            for i in range(0, len(df_numeric) - word_len + 1, word_len):
                if len(df_numeric[i:i+word_len]) == word_len:
                    word = df_numeric[i:i+word_len].values.reshape(-1)
                    test_targets_set.add(torch.tensor(word, dtype=torch.float32).unsqueeze(0).numpy().tobytes())

        test_pairs = [
            (x, y) for x, y in test_pairs_full
            if y.numpy().tobytes() in test_targets_set
        ]

        train_input, train_target = cls.tensorize_and_pad([(x, y) for x, y, _ in train_pairs], max_seq_len)
        val_input, val_target = cls.tensorize_and_pad([(x, y) for x, y, _ in val_pairs], max_seq_len)
        test_input, test_target = cls.tensorize_and_pad(test_pairs, max_seq_len)

        return cls(train_input, train_target, val_input, val_target, test_input, test_target)

    def summarize(self):
        names = ["Train", "Validation", "Test"]
        tensors = [
            (self.train_input, self.train_target),
            (self.val_input, self.val_target),
            (self.test_input, self.test_target),
        ]

        for name, (x, y) in zip(names, tensors):
            print(f"\n==== {name} Set ====")
            print(f"Input tensor shape: {x.shape}")
            print(f"Target tensor shape: {y.shape}")
            print(f"Batch size: {x.shape[0]}")
            print(f"Max sequence length: {x.shape[1]}")

            lengths = (x.abs().sum(dim=2) > 0).sum(dim=1).tolist()
            print(f"Min actual sequence length: {min(lengths)}")
            print(f"Max actual sequence length: {max(lengths)}")
            print(f"Avg actual sequence length: {sum(lengths) / len(lengths):.2f}")

            plt.hist(lengths, bins=range(min(lengths), max(lengths) + 1), alpha=0.6, label=name)

        plt.title("Distribution of Actual Sequence Lengths")
        plt.xlabel("Sequence Length")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return self

    def create_mask(self) -> dict:

        def compute_mask(x: torch.Tensor) -> torch.Tensor:
            return (x.abs().sum(dim=2) > 0).int()

        return {
            'train': compute_mask(self.train_input),
            'val': compute_mask(self.val_input),
            'test': compute_mask(self.test_input)
        }


def get_data_and_mask(
    dir_path: str,
    word_len: int = 7,
    features_per_hour: int = 5,
    max_seq_len: int = MAX_SEQ_LEN
) -> Tuple[dataTensors, dict]:
    """
    Load data from directory and create data tensors and masks.

    Args:
        dir_path (str): Path to the directory containing data files.
        word_len (int): Length of each word in the sequence.
        features_per_hour (int): Number of features per hour.
        max_seq_len (int): Maximum sequence length for padding.

    Returns:
        Tuple[dataTensors, dict]: Data tensors and masks for train, validation, and test sets.
    """
    dt = dataTensors.from_dir(dir_path, word_len=word_len, features_per_hour=features_per_hour, max_seq_len=max_seq_len)
    mask = dt.create_mask()
    return dt, mask



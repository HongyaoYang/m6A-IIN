"""Author-faithful m6A-IIN architecture adapted only at the regression head.

Architecture source of truth: ZLSRW/m6A-IIN commit
16a02193855b325538872b52df1331f88e32cea1. Deliberate task adaptations:
the final sigmoid is moved into stable Binomial NLL, and device-neutral tensor
operations replace statements that detach gradients or hard-code cuda:0.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import iv
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    sample_id: str
    sequence: str
    modified_count: int
    total_count: int


@dataclass(frozen=True)
class TrainConfig:
    sequence_length: int = 41
    folds: int = 1
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    attention_loss_weight: float = 0.0
    inverse_loss_weight: float = 0.0
    classification_threshold: float = 0.5
    coupling_layers: int = 1
    semantic_dim: int = 256
    hidden_dim: int = 271
    wavelet_scale: float = 0.3
    wavelet_threshold: float = 1e-4
    chebyshev_order: int = 3
    dropout: float = 0.2
    seed: int = 2023


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sequence_features(sequence: str) -> np.ndarray:
    """Exact 15-D nucleotide table released in 0.Structure-Graph.py."""
    table = {
        "A": [1, 0, 0, 0, 1, 1, 1, 0.2837, 0.3183, 0.1873, 0.3299, 0.2652, 0.3696, 0.1292, 0.2618],
        "C": [0, 1, 0, 0, 0, 1, 0, 0.2162, 0.2098, 0.2602, 0.2031, 0.2180, 0.1901, 0.2902, 0.2319],
        "G": [0, 0, 1, 0, 1, 0, 0, 0.2972, 0.2643, 0.2882, 0.2668, 0.2966, 0.2661, 0.2572, 0.2810],
        "U": [0, 0, 0, 1, 0, 0, 1, 0.2027, 0.2073, 0.2642, 0.2000, 0.2200, 0.1741, 0.3231, 0.2251],
    }
    return np.asarray([table[base] for base in sequence], dtype=np.float32)


def _author_match_brackets(structure: str) -> List[List[int]]:
    length = len(structure)
    matrix = [[0] * length for _ in range(length)]
    stack: List[int] = []
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise ValueError(f"Unmatched right parenthesis at index {index}")
            left = stack.pop()
            matrix[index][left] = matrix[left][index] = 1
        elif symbol != ".":
            raise ValueError(f"Unsupported dot-bracket symbol: {symbol!r}")
        matrix[index][index] = 1
    if stack:
        raise ValueError(f"Unmatched left parenthesis at index {stack[-1]}")
    return matrix


def _author_filter_elements(matrix: List[List[int]]) -> List[List[int]]:
    """Literal loop-matrix heuristic from the released preprocessing code."""
    rows = len(matrix)
    cols = len(matrix[0])
    filtered_matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    for xs in range(rows):
        flag = False
        i = xs
        j = xs + 1
        while i > 1:
            while j < cols - 2:
                if matrix[i][j] != 0 and matrix[i - 1][j + 1] != 0:
                    flag = True
                    i = 0
                    j = cols - 2
                j += 1
            i -= 1
        if flag:
            h = xs
            l = xs + 1
            while h > 0 and l < cols - 1:
                if matrix[h][l] == 0:
                    filtered_matrix[h][l] = 1
                h -= 1
                l += 1
    for xs in range(rows):
        h = xs
        l = xs + 1
        counter = 0
        while h > 1 and l <= cols - 2:
            if counter == 0 and filtered_matrix[h][l] != 0:
                start = h
                counter += 1
            elif counter > 0 and filtered_matrix[h][l] != 0:
                counter += 1
            elif counter > 0 and filtered_matrix[h][l] == 0:
                for i in range(start, start + counter):
                    for j in range(start, start + counter):
                        matrix[i][j] = 1
            h -= 1
            l += 1
    return filtered_matrix


def structure_matrices(
    structure: str,
    add_self_loops: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    adjacency = _author_match_brackets(structure)
    if not add_self_loops:
        for index in range(len(adjacency)):
            adjacency[index][index] = 0
    loop = _author_filter_elements(adjacency)
    return np.asarray(adjacency, dtype=np.float32), np.asarray(loop, dtype=np.float32)


class RNAfoldPredictor:
    """Compatibility marker; formal runs use precomputed RNAfold arrays."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("Formal protocol requires precomputed RNAfold structures")


class M6ARegressionDataset(Dataset):
    """Compatibility marker; the protocol uses its indexed lazy dataset."""

    def __init__(self, samples: Sequence[Sample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        raise RuntimeError("Use LazyM6ARegressionDataset from iin_protocol_common")


def normalize_and_symmetrize_tensor(
    input_tensor: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    # ``threshold`` is intentionally unused in the author's released utility.
    minimum = input_tensor.min(dim=-1, keepdim=True).values
    maximum = input_tensor.max(dim=-1, keepdim=True).values
    normalized = (input_tensor - minimum) / (maximum - minimum)
    normalized = normalized / (normalized.sum(dim=-1, keepdim=True) + 1e-8)
    return (normalized + normalized.transpose(-1, -2)) / 2.0


class CrossStructurePerception(nn.Module):
    """Author Cross_Structure_Perception (NCM implementation)."""

    def __init__(self, unit: int = 41) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(unit)
        self.bn1 = nn.BatchNorm1d(15)
        self.weight_key = nn.Parameter(torch.zeros(unit, unit))
        self.weight_query = nn.Parameter(torch.zeros(unit, unit))
        self.weight_value = nn.Parameter(torch.zeros(unit, unit))
        for parameter in (self.weight_key, self.weight_query, self.weight_value):
            nn.init.xavier_uniform_(parameter, gain=1.414)

    def forward(self, primary: torch.Tensor) -> torch.Tensor:
        primary = self.bn1(primary.permute(0, 2, 1)).permute(0, 2, 1)
        correlation = primary @ primary.transpose(1, 2)
        key = correlation @ self.weight_key
        query = correlation @ self.weight_query
        value = correlation @ self.weight_value
        data = query @ key.transpose(1, 2)
        data = F.leaky_relu(self.bn(data))
        coefficient = F.softmax(data, dim=2)
        perception = coefficient @ correlation @ value
        perception = normalize_and_symmetrize_tensor(perception)
        return 0.5 * (perception + perception.transpose(1, 2))


class LoopLearning(nn.Module):
    """Author loop_learning (LRP implementation), including sigma=0.5."""

    def __init__(self, unit: int = 41) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(15)
        self.weight_loop = nn.Parameter(torch.zeros(unit, unit))
        nn.init.xavier_uniform_(self.weight_loop, gain=1.414)
        self.loop_representation = nn.Sequential(
            nn.Linear(41, 41),
            nn.Tanh(),
            nn.Linear(41, 15),
        )

    def forward(self, loop_adjacency: torch.Tensor) -> torch.Tensor:
        symmetric_loop = loop_adjacency + loop_adjacency.transpose(1, 2)
        filtered_matrix = self.weight_loop * symmetric_loop
        loop_matrix = torch.exp(-0.5 * (filtered_matrix / 0.5).square())
        projected = self.loop_representation(loop_matrix)
        return self.bn1(projected.permute(0, 2, 1)).permute(0, 2, 1)


class GraphCouplingLayer(nn.Module):
    def __init__(self, size: int = 41) -> None:
        super().__init__()
        self.informationLoop = LoopLearning(size)
        self.informationTrans = CrossStructurePerception(size)

    def forward(
        self,
        adjacency: torch.Tensor,
        loop_adjacency: torch.Tensor,
        primary: torch.Tensor,
        invertible: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not invertible:
            coupled_primary = primary + self.informationLoop(loop_adjacency)
            coupled_adjacency = adjacency + self.informationTrans(coupled_primary)
            # Minimal executable correction: author torch.tensor(tensor)
            # detaches gradients, while these are the same values without detach.
            return coupled_adjacency, coupled_primary
        initial_adjacency = adjacency - self.informationTrans(primary)
        initial_primary = primary - self.informationLoop(loop_adjacency)
        return initial_adjacency, initial_primary


class GraphInteraction(nn.Module):
    def __init__(self, num: int = 1, size: int = 41) -> None:
        super().__init__()
        coupling = GraphCouplingLayer(size)
        self.layers = nn.ModuleList([coupling for _ in range(num)])

    def forward(
        self,
        adjacency: torch.Tensor,
        loop_adjacency: torch.Tensor,
        primary: torch.Tensor,
        invertible: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_adjacency = adjacency
        output_primary = primary
        for layer in self.layers:
            # Literal released loop: each layer receives the original inputs.
            output_adjacency, output_primary = layer(
                adjacency, loop_adjacency, primary, invertible
            )
        return output_adjacency, output_primary


class AuthorSemanticNetwork(nn.Module):
    """Released ELMom6A.Semantic_network topology.

    Input is [15-D coupled primary || 256-D precomputed Flair ELMo]. The two
    LSTMs are separate unidirectional modules and their outputs are averaged,
    matching the public code instead of rewriting it as a new BiLSTM.
    """

    def __init__(self, drop_rate: float = 0.2, filters1: int = 271, filters2: int = 271) -> None:
        super().__init__()
        conv_weights = torch.clamp(torch.rand(256, 4, 5), 0, 1)
        self.conv1d_layer = nn.Conv1d(4, 256, padding=2, kernel_size=5)
        self.conv1d_layer.weight.data.copy_(conv_weights)
        self.conv1 = nn.Conv1d(271, filters1, kernel_size=5, padding=2)
        self.batchnorm1 = nn.BatchNorm1d(filters1)
        self.conv2 = nn.Conv1d(filters1, filters2, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool1d(kernel_size=1)
        self.dropout1 = nn.Dropout(drop_rate)
        self.conv3 = nn.Conv1d(271, filters1, kernel_size=5, padding=2)
        self.batchnorm2 = nn.BatchNorm1d(filters1)
        self.conv4 = nn.Conv1d(filters1, filters2, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=1)
        self.dropout2 = nn.Dropout(drop_rate)
        self.lstm1 = nn.LSTM(filters2, 271, bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(filters2, 271, bidirectional=False, batch_first=True)
        self.dropout3 = nn.Dropout(drop_rate)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.permute(0, 2, 1)
        out1 = torch.tanh(self.conv1(x))
        out1 = self.batchnorm1(out1)
        out1 = self.dropout1(out1)
        out1 = F.relu(self.conv2(out1))
        out1 = self.maxpool1(out1)
        out1 = self.dropout1(out1)
        outx, _ = self.lstm1(out1.permute(0, 2, 1))
        outy, _ = self.lstm2(out1.permute(0, 2, 1))
        return self.dropout3((outx + outy) / 2.0)


class AuthorFastWaveletBasis(nn.Module):
    """Released fast_wavelet_basis (scale=.3, threshold=1e-4, order=3)."""

    def __init__(self, scale: float = 0.3, threshold: float = 1e-4, order: int = 3) -> None:
        super().__init__()
        self.scale = scale
        self.threshold = threshold
        self.order = order

    @staticmethod
    def laplacian_multi(adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1)
        symmetric = 0.5 * (adjacency + adjacency.transpose(1, 2))
        degree_matrix = torch.diag_embed(degree)
        inv_sqrt_matrix = torch.diag_embed(1.0 / (torch.sqrt(degree) + 1e-6))
        return inv_sqrt_matrix @ (degree_matrix - symmetric) @ inv_sqrt_matrix

    @staticmethod
    def _largest_lambdas(laplacian: torch.Tensor) -> np.ndarray:
        matrices = laplacian.detach().cpu().numpy()
        values = []
        for matrix in matrices:
            try:
                eigenvalues, _ = eigsh(matrix, k=1, which="LM")
                largest = eigenvalues[0]
            except ArpackNoConvergence:
                # ARPACK can very occasionally fail on small matrices with
                # repeated or tightly clustered eigenvalues.  Preserve the
                # author's eigsh path normally, and compute the same largest
                # eigenvalue exactly only for the matrix that failed.
                warnings.warn(
                    "ARPACK did not converge; using dense symmetric eigvalsh "
                    "for the same largest-eigenvalue calculation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                symmetric = 0.5 * (matrix + matrix.T)
                largest = np.linalg.eigvalsh(symmetric)[-1]
            if not np.isfinite(largest):
                raise FloatingPointError("Non-finite largest Laplacian eigenvalue")
            values.append(largest)
        return np.asarray(values)

    @staticmethod
    def _chebyshev_multi(laplacian: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = laplacian.shape
        expanded = laplacian.unsqueeze(1)
        first = torch.zeros(
            batch, 1, nodes, nodes, device=laplacian.device, dtype=torch.float32
        )
        second = expanded
        third = 2 * (expanded @ second) - first
        fourth = 2 * (expanded @ third) - second
        return torch.cat([first, second, third, fourth], dim=1)

    def forward(self, adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        laplacian = self.laplacian_multi(adjacency)
        a = self._largest_lambdas(laplacian) / 2.0
        coefficients = [
            [2 * np.exp(self.scale * value) * iv(i, self.scale * value)
             for i in range(self.order + 1)]
            for value in a
        ]
        inverse_coefficients = [
            [2 * np.exp(-self.scale * value) * iv(i, -self.scale * value)
             for i in range(self.order + 1)]
            for value in a
        ]
        coefficients_t = torch.as_tensor(
            np.asarray(coefficients), device=adjacency.device
        ).unsqueeze(-1).unsqueeze(-1)
        inverse_t = torch.as_tensor(
            np.asarray(inverse_coefficients), device=adjacency.device
        ).unsqueeze(-1).unsqueeze(-1)
        chebyshev = self._chebyshev_multi(laplacian)
        wavelet = torch.sum(coefficients_t * chebyshev, dim=1)
        inverse = torch.sum(inverse_t * chebyshev, dim=1)
        wavelet = torch.where(wavelet < self.threshold, 0.0, wavelet)
        inverse = torch.where(inverse < self.threshold, 0.0, inverse)
        wavelet = F.normalize(wavelet, p=1, dim=2)
        inverse = F.normalize(inverse, p=1, dim=2)
        return wavelet.float(), inverse.float()


class GWNNLayer(nn.Module):
    def __init__(self, node_num: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.weight_matrix = nn.Parameter(torch.empty(in_channels, out_channels))
        self.filter = nn.Parameter(torch.empty(node_num))
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight_matrix)

    def forward(
        self,
        features: torch.Tensor,
        wavelets: torch.Tensor,
        wavelets_inv: torch.Tensor,
    ) -> torch.Tensor:
        transformed = features @ self.weight_matrix
        return wavelets @ (torch.diag(self.filter) @ (wavelets_inv @ transformed))


class GraphWaveletNeuralNetwork(nn.Module):
    def __init__(self, node_cnt: int = 41, feature_dims: int = 271, hidden_dims: int = 271, dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.conv_1 = GWNNLayer(node_cnt, feature_dims, hidden_dims)
        self.conv_2 = GWNNLayer(node_cnt, hidden_dims, hidden_dims)

    def forward(self, features: torch.Tensor, wavelets: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
        first = F.dropout(
            F.relu(self.conv_1(features, wavelets, inverse)),
            training=self.training,
            p=self.dropout_rate,
        )
        return self.conv_2(first, wavelets, inverse)


class ScalingLayer(nn.Module):
    def __init__(self, data_dim: int = 271) -> None:
        super().__init__()
        self.log_scale_vector = nn.Parameter(torch.randn(41, data_dim))

    def forward(self, representation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coefficient = torch.exp(self.log_scale_vector)
        return coefficient.unsqueeze(0) * representation, coefficient


class M6AIINForwardRegressor(nn.Module):
    """Released InvertibleBigraphNeuralNetwork with a scalar-logit head."""

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        if config.sequence_length != 41:
            raise ValueError("Author architecture is fixed to 41 nucleotides")
        if config.semantic_dim != 256 or config.hidden_dim != 271:
            raise ValueError("Author architecture requires ELMo=256 and hidden=271")
        self.config = config
        self.graph_Interaction = GraphInteraction(config.coupling_layers, 41)
        self.bn = nn.BatchNorm1d(41)
        self.bnSR = nn.BatchNorm1d(15)
        self.bn1 = nn.BatchNorm1d(271)
        self.scalingLayer = ScalingLayer(271)
        self.graph_wavelet_layer = GraphWaveletNeuralNetwork(41, 271, 271, config.dropout)
        self.shape1 = nn.Sequential(
            nn.Linear(271, 271),
            nn.ReLU(),
            nn.Linear(271, 271),
        )
        self.scalingWeight = nn.Sequential(nn.Linear(41, 41), nn.Sigmoid())
        self.scalingWeight_SR = nn.Sequential(nn.Linear(15, 15), nn.Sigmoid())
        self.prob_linear = nn.Linear(271, 1)
        self.gru = nn.GRU(271, 271)  # released but unused
        self.semantics = AuthorSemanticNetwork(config.dropout)
        self.wavelet_basis = AuthorFastWaveletBasis(
            config.wavelet_scale,
            config.wavelet_threshold,
            config.chebyshev_order,
        )

    def forward(
        self,
        adjacency: torch.Tensor,
        loop_adjacency: torch.Tensor,
        primary: torch.Tensor,
        elmo_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_output, structural_representation = self.graph_Interaction(
            adjacency, loop_adjacency, primary, False
        )
        wavelets, wavelet_inverse = self.wavelet_basis(graph_output)
        combined = torch.cat([structural_representation, elmo_features], dim=2).float()
        semantic = self.semantics(combined)
        final_representation = self.graph_wavelet_layer(
            semantic, wavelets, wavelet_inverse
        )
        final_representation, scaling_coefficient = self.scalingLayer(final_representation)
        scaling_weight = scaling_coefficient @ scaling_coefficient.transpose(0, 1)
        scaling_weight = self.scalingWeight(self.bn(scaling_weight))
        scaling_weight = (scaling_weight + scaling_weight.transpose(0, 1)) / 2.0
        scaling_weight = torch.where(scaling_weight < 0.5, 0.0, scaling_weight)
        scaling_coefficient_sr = self.scalingWeight_SR(
            self.bnSR(scaling_coefficient[:, :15])
        )
        inductive_graph = scaling_weight.unsqueeze(0) * graph_output
        reconstructed_graph, reconstructed_primary = self.graph_Interaction(
            inductive_graph,
            loop_adjacency,
            scaling_coefficient_sr.unsqueeze(0) * structural_representation,
            True,
        )
        shaped = self.shape1(final_representation)
        sequence_feature = self.bn1(torch.mean(shaped, dim=1))
        logits = self.prob_linear(sequence_feature.unsqueeze(1)).squeeze(-1).squeeze(-1)
        return logits, scaling_weight, reconstructed_graph, reconstructed_primary, graph_output


class BinomialCountNLLLoss(nn.Module):
    def forward(self, logits: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return (m * F.softplus(logits) - y * logits).mean()


def inverse_reconstruction_loss(
    reconstructed_adjacency: torch.Tensor,
    adjacency: torch.Tensor,
    reconstructed_primary: torch.Tensor,
    primary: torch.Tensor,
) -> torch.Tensor:
    batch_size = adjacency.shape[0]
    return (
        F.mse_loss(reconstructed_adjacency, adjacency, reduction="sum")
        + F.mse_loss(reconstructed_primary, primary, reduction="sum")
    ) / (2.0 * batch_size)


def regional_attention_loss(scaling_weight: torch.Tensor, batch_size: int) -> torch.Tensor:
    return scaling_weight.sum() / batch_size


def move_batch(batch, device: torch.device):
    primary, adjacency, loop_adjacency, elmo, y, m, sample_ids = batch
    return (
        primary.to(device, non_blocking=True),
        adjacency.to(device, non_blocking=True),
        loop_adjacency.to(device, non_blocking=True),
        elmo.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
        m.to(device, non_blocking=True),
        sample_ids,
    )

#!/usr/bin/env python3
"""Small CPU/GPU smoke test for the audited author-faithful IIN implementation."""

from __future__ import annotations

import numpy as np
import torch

from m6a_iin_forward_regression import (
    BinomialCountNLLLoss,
    M6AIINForwardRegressor,
    TrainConfig,
    sequence_features,
    structure_matrices,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences = [("ACGU" * 11)[:41], ("UGCA" * 11)[:41]]
    structures = ["((((....))))" + "." * 29, "." * 5 + "(((.....)))" + "." * 25]
    assert all(len(value) == 41 for value in sequences + structures)

    primary = torch.from_numpy(
        np.stack([sequence_features(value) for value in sequences])
    ).to(device)
    graph_pairs = [structure_matrices(value) for value in structures]
    adjacency = torch.from_numpy(np.stack([value[0] for value in graph_pairs])).to(device)
    loops = torch.from_numpy(np.stack([value[1] for value in graph_pairs])).to(device)
    elmo = torch.randn(2, 41, 256, device=device)

    model = M6AIINForwardRegressor(TrainConfig()).to(device)
    logits, scaling, reconstructed_graph, reconstructed_primary, _ = model(
        adjacency, loops, primary, elmo
    )
    loss = BinomialCountNLLLoss()(
        logits,
        torch.tensor([2.0, 7.0], device=device),
        torch.tensor([10.0, 10.0], device=device),
    )
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite Binomial NLL")
    loss.backward()

    expected_shapes = {
        "logits": ((2,), logits.shape),
        "scaling": ((41, 41), scaling.shape),
        "reconstructed_graph": ((2, 41, 41), reconstructed_graph.shape),
        "reconstructed_primary": ((2, 41, 15), reconstructed_primary.shape),
    }
    for name, (expected, actual) in expected_shapes.items():
        if tuple(actual) != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {tuple(actual)}")

    required_gradients = (
        "graph_Interaction.layers.0.informationLoop.weight_loop",
        "graph_Interaction.layers.0.informationTrans.weight_key",
        "graph_wavelet_layer.conv_1.weight_matrix",
        "prob_linear.weight",
    )
    parameters = dict(model.named_parameters())
    for name in required_gradients:
        gradient = parameters[name].grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"Missing or invalid gradient: {name}")

    print(f"author-faithful IIN smoke test: PASS ({device})")
    print(f"Binomial NLL: {float(loss):.6f}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()

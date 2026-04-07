"""GCN connection pruner: predict which connections are in the optimal hyperpath.

Uses a Graph Convolutional Network on the transit bipartite graph
(connections + stops) to predict P(c in H*) for each connection.
Connections with low probability are pruned, and Durner's exact Bellman
runs on the pruned (smaller) graph.

Follows Nair et al. (2020) Neural Diving principle and
Tang et al. (2019) Learning to Cut paradigm.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

import torch
import torch.nn as nn

from ..transit_graph import TransitGraph


class SimplePruner(nn.Module):
    """MLP-based connection pruner (simpler alternative to full GCN).

    For the synthetic network, a GCN is overkill. This MLP takes
    per-connection features and predicts P(c in optimal hyperpath).
    Can be upgraded to GCN (torch_geometric) when needed.
    """

    def __init__(self, input_dim: int = 12, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ConnectionPruner:
    """Wrapper: trains and uses a pruner to select relevant connections.

    Training: from Durner's exact solutions, label each connection as
    in/not-in the optimal hyperpath. Train binary classifier.

    Inference: predict P(c in H*), keep connections above threshold.
    Run exact Bellman on pruned set.
    """

    def __init__(self, input_dim: int = 12, hidden_dim: int = 32):
        self.model = SimplePruner(input_dim, hidden_dim)
        self.trained = False

    def train_pruner(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 80,
        lr: float = 1e-3,
        verbose: bool = True,
    ):
        """Train the pruner.

        Args:
            X: (N, feature_dim) connection features.
            y: (N,) binary labels (1 = in hyperpath, 0 = not).
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        # Handle class imbalance (most connections not in hyperpath)
        pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(min(pos_weight, 10.0)))
        # Switch model to use raw logits
        self.model.net[-1] = nn.Identity()

        self.model.train()
        for epoch in range(epochs):
            pred = self.model(X_t)
            loss = loss_fn(pred, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 20 == 0:
                with torch.no_grad():
                    probs = torch.sigmoid(pred)
                    pred_labels = (probs > 0.5).float()
                    acc = (pred_labels == y_t).float().mean()
                    recall = (pred_labels[y_t == 1] == 1).float().mean() if y_t.sum() > 0 else 0
                    print(f"  Epoch {epoch+1}: loss={loss.item():.4f} acc={acc:.3f} recall={recall:.3f}")

        # Restore sigmoid for inference
        self.model.net[-1] = nn.Sigmoid()
        self.model.eval()
        self.trained = True

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Predict P(c in H*) for each connection."""
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            probs = self.model(X_t).numpy()
        return probs

    def prune(
        self,
        X: np.ndarray,
        connection_ids: list[int],
        threshold: float = 0.3,
    ) -> list[int]:
        """Return connection IDs predicted to be in hyperpath.

        Args:
            X: (N, feature_dim) features for each connection.
            connection_ids: Corresponding connection IDs.
            threshold: Keep connections with P > threshold.

        Returns:
            Pruned list of connection IDs.
        """
        probs = self.predict_probabilities(X)
        kept = [cid for cid, p in zip(connection_ids, probs) if p > threshold]
        return kept

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
        self.trained = True

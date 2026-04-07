"""V-hat ensemble: neural value network for fast arrival time prediction.

Replaces Durner's O(|C|^2 · |T|) Bellman PMF propagation with
O(|C|) neural evaluations. Each network in the ensemble predicts
E[arrival_time | stop, time, regime]. Ensemble disagreement (std)
serves as an uncertainty estimate.

Training data comes from Durner's exact solutions (offline oracle).
This follows the Cappart (2020) principle: RL value function as a
fast surrogate for expensive DP.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """Single MLP predicting expected arrival time."""

    def __init__(self, input_dim: int = 14, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ValueEnsemble:
    """Ensemble of K value networks with uncertainty estimation.

    Predicts E[arrival] and std(arrival) from (stop, time, regime) features.
    High std → low confidence → may trigger fallback to exact Durner.
    """

    def __init__(self, input_dim: int = 14, hidden_dim: int = 64, n_models: int = 5):
        self.n_models = n_models
        self.models = [ValueNetwork(input_dim, hidden_dim) for _ in range(n_models)]
        self.trained = False

    def train_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        verbose: bool = True,
    ):
        """Train all models with bootstrap sampling (different data subsets)."""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        n_samples = len(X)

        for k, model in enumerate(self.models):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            # Bootstrap: sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_tensor[indices]
            y_boot = y_tensor[indices]

            model.train()
            for epoch in range(epochs):
                perm = torch.randperm(n_samples)
                total_loss = 0.0
                n_batches = 0
                for i in range(0, n_samples, batch_size):
                    batch_idx = perm[i:i + batch_size]
                    x_batch = X_boot[batch_idx]
                    y_batch = y_boot[batch_idx]

                    pred = model(x_batch)
                    loss = loss_fn(pred, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1

                if verbose and (epoch + 1) % 20 == 0:
                    print(f"  Model {k}, epoch {epoch+1}: loss={total_loss/n_batches:.4f}")

            model.eval()

        self.trained = True

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """Predict mean arrival time and uncertainty for a single input.

        Args:
            features: (feature_dim,) array.

        Returns:
            (mean_prediction, std_prediction)
        """
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        preds = []
        with torch.no_grad():
            for model in self.models:
                preds.append(model(x).item())
        preds = np.array(preds)
        return float(preds.mean()), float(preds.std())

    def predict_batch(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict for a batch of inputs.

        Args:
            features: (N, feature_dim) array.

        Returns:
            (means, stds) each of shape (N,)
        """
        x = torch.tensor(features, dtype=torch.float32)
        all_preds = []
        with torch.no_grad():
            for model in self.models:
                all_preds.append(model(x).numpy())
        all_preds = np.stack(all_preds, axis=0)  # (K, N)
        return all_preds.mean(axis=0), all_preds.std(axis=0)

    def save(self, path: str):
        state = {f"model_{k}": m.state_dict() for k, m in enumerate(self.models)}
        torch.save(state, path)

    def load(self, path: str):
        state = torch.load(path, weights_only=True)
        for k, m in enumerate(self.models):
            m.load_state_dict(state[f"model_{k}"])
            m.eval()
        self.trained = True

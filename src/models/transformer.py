"""Minimal Transformer encoder for sequence-to-sequence regression."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class _WindowDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.as_tensor(inputs, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - exercised indirectly
        length = x.size(1)
        return x + self.pe[:, :length]


class _TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        horizon: int,
        target_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = _PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, horizon * target_dim)
        self.horizon = horizon
        self.target_dim = target_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - exercised indirectly
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        encoded = self.encoder(x)
        pooled = torch.mean(encoded, dim=1)
        preds = self.head(pooled)
        return preds.view(-1, self.horizon, self.target_dim)


@dataclass
class TransformerModel:
    d_model: int
    nhead: int
    num_layers: int
    dropout: float = 0.0
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    patience: int = 5

    _net: Optional[nn.Module] = field(init=False, default=None)
    _input_dim: Optional[int] = field(init=False, default=None)
    _target_dim: int = field(init=False, default=1)
    _horizon: int = field(init=False, default=1)
    _squeeze_output: bool = field(init=False, default=False)
    _state_dict: Optional[Dict[str, torch.Tensor]] = field(init=False, default=None)
    metrics: Dict[str, float] = field(init=False, default_factory=dict)

    def _ensure_network(self, input_dim: int) -> None:
        if self._net is None:
            self._net = _TransformerRegressor(
                input_dim=input_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                horizon=self._horizon,
                target_dim=self._target_dim,
                dropout=self.dropout,
            )

    def fit(
        self,
        train_data: Any,
        val_data: Any | None = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, float]:
        if loss_fn is None:
            loss_fn = lambda pred, target: torch.mean((pred - target) ** 2)

        inputs = np.asarray(train_data.inputs, dtype=np.float32)
        targets = np.asarray(train_data.targets, dtype=np.float32)
        if targets.ndim == 2:
            targets = targets[..., None]
            self._squeeze_output = True
        else:
            self._squeeze_output = False
        self._horizon = targets.shape[1]
        self._target_dim = targets.shape[2]
        self._input_dim = inputs.shape[-1]
        self._ensure_network(self._input_dim)

        device = torch.device("cpu")
        self._net.to(device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)

        train_loader = DataLoader(
            _WindowDataset(inputs, targets),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
        if val_data is not None:
            val_inputs = np.asarray(val_data.inputs, dtype=np.float32)
            val_targets = np.asarray(val_data.targets, dtype=np.float32)
            if val_targets.ndim == 2:
                val_targets = val_targets[..., None]
            val_loader = DataLoader(
                _WindowDataset(val_inputs, val_targets),
                batch_size=self.batch_size,
                shuffle=False,
            )

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val = float("inf")
        patience_left = self.patience
        last_train = float("inf")

        for _ in range(self.epochs):
            self._net.train()
            total = 0.0
            count = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                preds = self._net(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                total += float(loss.item()) * xb.size(0)
                count += xb.size(0)
            last_train = total / max(count, 1)

            if val_loader is not None:
                self._net.eval()
                with torch.no_grad():
                    val_total = 0.0
                    val_count = 0
                    for xb, yb in val_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        preds = self._net(xb)
                        loss = loss_fn(preds, yb)
                        val_total += float(loss.item()) * xb.size(0)
                        val_count += xb.size(0)
                val_loss = val_total / max(val_count, 1)
            else:
                val_loss = last_train

            if val_loss + 1e-6 < best_val or best_state is None:
                best_val = val_loss
                best_state = copy.deepcopy(self._net.state_dict())
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is None:
            best_state = copy.deepcopy(self._net.state_dict())
        self._net.load_state_dict(best_state)
        self._state_dict = best_state
        self.metrics = {"train_loss": last_train, "val_loss": best_val}
        return self.metrics

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model must be fitted before calling predict")
        arr = np.asarray(inputs, dtype=np.float32)
        tensor = torch.as_tensor(arr, dtype=torch.float32)
        self._net.eval()
        with torch.no_grad():
            preds = self._net(tensor).cpu().numpy()
        if self._squeeze_output:
            preds = preds.squeeze(-1)
        return preds

    def state_dict(self) -> Dict[str, torch.Tensor]:
        if self._state_dict is None and self._net is not None:
            self._state_dict = copy.deepcopy(self._net.state_dict())
        if self._state_dict is None:
            raise RuntimeError("State dict requested before training")
        return self._state_dict

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        if self._net is None:
            if self._input_dim is None:
                raise RuntimeError("Input dimension unknown; fit the model before loading state")
            self._ensure_network(self._input_dim)
        if self._net is None:
            raise RuntimeError("Network not initialised")
        self._net.load_state_dict(state)
        self._state_dict = copy.deepcopy(state)

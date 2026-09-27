"""CUDA quantile MLP and 8-quarter GRU challengers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from treasuryshield.development.models import FitResult
from treasuryshield.development.preprocess import NumericPreprocessor
from treasuryshield.splits.temporal import QUANTILES, deterministic_sample

SEQUENCE_FEATURES = [
    "log_deposits", "cash_asset_ratio", "liquid_broad_asset_ratio",
    "loan_deposit_ratio", "borrowings_asset_ratio", "capital_asset_ratio",
    "brokered_domestic_ratio", "deposit_cost_annualized", "deposit_growth_qoq_lagged",
    "macro_DFF_revised", "macro_VIXCLS_revised",
]


class QuantileHead(nn.Module):
    def __init__(self, width: int, monotonic: bool):
        super().__init__()
        self.linear = nn.Linear(width, len(QUANTILES))
        self.monotonic = monotonic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.linear(x)
        if not self.monotonic:
            return raw
        first = raw[:, :1]
        increments = torch.nn.functional.softplus(raw[:, 1:])
        return torch.cat([first, first + torch.cumsum(increments, dim=1)], dim=1)


class MLPQuantile(nn.Module):
    def __init__(self, input_dim: int, width: int = 128, monotonic: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(.1),
            nn.Linear(width, width // 2), nn.GELU(), nn.Dropout(.1))
        self.head = QuantileHead(width // 2, monotonic)

    def forward(self, x: torch.Tensor, sequence: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.encoder(x))


class GRUQuantile(nn.Module):
    def __init__(self, input_dim: int, sequence_dim: int, width: int = 96,
                 monotonic: bool = False):
        super().__init__()
        self.static = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
        self.gru = nn.GRU(sequence_dim, width, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Dropout(.1))
        self.head = QuantileHead(width, monotonic)

    def forward(self, x: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(sequence)
        fused = self.fusion(torch.cat([self.static(x), hidden[-1]], dim=1))
        return self.head(fused)


def _sequence_array(frame: pd.DataFrame, history: pd.DataFrame,
                    pre: NumericPreprocessor, names: list[str], length: int) -> np.ndarray:
    if history.duplicated(["bank_id", "stata_quarter"]).any():
        raise ValueError("Duplicate bank-quarter in sequence history")
    lookup = history.set_index(["bank_id", "stata_quarter"])[names]
    bank = frame["bank_id"].to_numpy()
    quarter = frame["stata_quarter"].to_numpy()
    pieces = []
    for lag in range(length - 1, -1, -1):
        keys = pd.MultiIndex.from_arrays([bank, quarter - lag], names=["bank_id", "stata_quarter"])
        slice_frame = lookup.reindex(keys).reset_index(drop=True)
        pieces.append(pre.transform(slice_frame))
    return np.stack(pieces, axis=1)


def _predict(model: nn.Module, x: np.ndarray, sequence: np.ndarray | None,
             device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], device=device)
            sb = (torch.as_tensor(sequence[start:start + batch_size], device=device)
                  if sequence is not None else None)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(xb, sb)
            results.append(output.float().cpu().numpy())
    return np.vstack(results) if results else np.empty((0, len(QUANTILES)))


def train_neural(train: pd.DataFrame, calibration: pd.DataFrame,
                 test: pd.DataFrame, names: list[str], history: pd.DataFrame,
                 *, architecture: str = "mlp", monotonic: bool = False,
                 seed: int = 17, epochs: int = 18, max_training_rows: int = 200_000,
                 width: int = 128, batch_size: int = 4096,
                 tail_weighted: bool = False) -> FitResult:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-required neural model cannot run without available CUDA")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    sampled = deterministic_sample(train, max_training_rows, seed)
    static_pre = NumericPreprocessor(names, linear=True, clip_neural=True).fit(sampled)
    train_x = static_pre.transform(sampled)
    calibration_x = static_pre.transform(calibration)
    test_x = static_pre.transform(test)
    seq_pre = None
    train_seq = calibration_seq = test_seq = None
    if architecture == "gru":
        sequence_names = [name for name in SEQUENCE_FEATURES if name in history]
        seq_pre = NumericPreprocessor(sequence_names, linear=True).fit(sampled)
        train_seq = _sequence_array(sampled, history, seq_pre, sequence_names, 8)
        calibration_seq = _sequence_array(calibration, history, seq_pre, sequence_names, 8)
        test_seq = _sequence_array(test, history, seq_pre, sequence_names, 8)
        model: nn.Module = GRUQuantile(train_x.shape[1], train_seq.shape[2],
                                       width=max(64, width // 2), monotonic=monotonic)
    elif architecture == "mlp":
        model = MLPQuantile(train_x.shape[1], width=width, monotonic=monotonic)
    else:
        raise ValueError(architecture)
    if monotonic:
        # A softplus head initialized near zero would start with ~0.69 log
        # points between adjacent quantiles. Initialize from historical
        # training quantiles so the constrained model starts on the target
        # scale, without using calibration or evaluation labels.
        initial = np.quantile(sampled["deposit_log_growth"].to_numpy(), QUANTILES)
        increments = np.maximum(np.diff(initial), 1e-5)
        bias = np.concatenate(([initial[0]], np.log(np.expm1(increments))))
        with torch.no_grad():
            model.head.linear.weight.zero_()
            model.head.linear.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    device = torch.device("cuda")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda")
    quantiles = torch.tensor(QUANTILES, device=device, dtype=torch.float32)[None, :]
    quantile_weights = torch.tensor([2.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0]
                                    if tail_weighted else [1.0] * len(QUANTILES),
                                    device=device, dtype=torch.float32)[None, :]
    y = sampled["deposit_log_growth"].to_numpy(dtype=np.float32)
    fallback = []
    current_batch = batch_size
    for epoch in range(epochs):
        model.train()
        permutation = np.random.default_rng(seed + epoch).permutation(len(sampled))
        position = 0
        while position < len(permutation):
            indices = permutation[position:position + current_batch]
            try:
                xb = torch.as_tensor(train_x[indices], device=device)
                yb = torch.as_tensor(y[indices], device=device)[:, None]
                sb = torch.as_tensor(train_seq[indices], device=device) if train_seq is not None else None
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = model(xb, sb)
                    error = yb - prediction
                    loss = (torch.maximum(quantiles * error, (quantiles - 1) * error) *
                            quantile_weights).sum(dim=1).mean() / quantile_weights.sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite neural pinball loss")
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                position += len(indices)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if current_batch <= 256:
                    raise
                current_batch //= 2
                fallback.append({"epoch": epoch, "reduced_batch_size": current_batch})
    calibration_pred = _predict(model, calibration_x, calibration_seq, device)
    test_pred = _predict(model, test_x, test_seq, device)
    return FitResult(calibration_pred, test_pred,
                     {"model": model, "static_preprocessor": static_pre,
                      "sequence_preprocessor": seq_pre},
                     {"training_rows": len(sampled), "architecture": architecture,
                      "monotonic": monotonic, "epochs": epochs,
                      "tail_weighted": tail_weighted,
                      "device": torch.cuda.get_device_name(0),
                      "final_batch_size": current_batch, "oom_fallbacks": fallback})

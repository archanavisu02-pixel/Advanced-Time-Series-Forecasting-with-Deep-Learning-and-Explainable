"""
Advanced Time Series Forecasting with Deep Learning and Explainability
----------------------------------------------------------------------
Single-file production-quality Python script that:
 - Generates a complex non-stationary synthetic time series
 - Implements a Holt–Winters baseline (ExponentialSmoothing)
 - Implements LSTM and Transformer regressors in PyTorch
 - Uses rolling-origin CV with robust checks (no more "need at least one array to stack" errors)
 - Evaluates using RMSE, MAE, MAPE
 - Provides explainability scaffolds (SHAP for LSTM, attention extraction for Transformer)
 - Saves outputs to ./outputs/

Usage: run this file with Python 3.8+. GPU optional but helpful.
Requirements:
 pip install numpy pandas matplotlib torch scikit-learn statsmodels shap tqdm

"""

import os
import math
import random
import pickle
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Optional: SHAP (import only when used)
try:
    import shap
except Exception:
    shap = None

# -------------------------------
# Reproducibility & device
# -------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Utilities: metrics
# -------------------------------

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape_v = mape(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape_v}

# -------------------------------
# 1) Data generation (complex non-stationary)
# -------------------------------

def generate_complex_time_series(n_steps: int = 3000, seed: int = SEED) -> pd.DataFrame:
    np.random.seed(seed)
    t = np.arange(n_steps)

    # Trend
    trend = 0.005 * (t - n_steps / 2) + 0.00001 * (t - n_steps / 2) ** 2

    # Seasonalities
    season_daily = 2.0 * np.sin(2 * np.pi * t / 24.0)
    season_weekly = 1.5 * np.sin(2 * np.pi * t / (24.0 * 7.0))
    season_yearly = 0.5 * np.sin(2 * np.pi * t / (24.0 * 365.0))

    # Regime piecewise
    regime = np.where(t < n_steps * 0.4, 1.0, np.where(t < n_steps * 0.7, 1.6, 0.7))

    # AR-like component
    ar = np.zeros(n_steps)
    for i in range(2, n_steps):
        ar[i] = 0.6 * ar[i - 1] - 0.2 * ar[i - 2] + 0.1 * np.sin(0.1 * i)

    # Heteroscedastic noise
    noise_std = 0.3 + 0.5 * np.sin(2 * np.pi * t / (24.0 * 30.0))
    noise = regime * noise_std * np.random.randn(n_steps)

    series = 10 + trend + season_daily + season_weekly + season_yearly + ar + noise

    dates = pd.date_range(start='2018-01-01', periods=n_steps, freq='H')
    df = pd.DataFrame({"ds": dates, "y": series})
    return df

# -------------------------------
# Feature engineering
# -------------------------------

def create_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    df_ = df.copy()
    for lag in lags:
        df_[f"lag_{lag}"] = df_["y"].shift(lag)

    df_["rolling_mean_24"] = df_["y"].rolling(window=24, min_periods=1).mean().shift(1)
    df_["rolling_std_24"] = df_["y"].rolling(window=24, min_periods=1).std().shift(1).fillna(0)

    # time features
    df_["hour"] = df_["ds"].dt.hour
    df_["dayofweek"] = df_["ds"].dt.dayofweek

    df_.dropna(inplace=True)
    return df_

# -------------------------------
# Dataset class
# -------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------------
# Models
# -------------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out_last = out[:, -1, :]
        return self.fc(out_last).squeeze(-1)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))

    def forward(self, x):
        x = self.input_proj(x)
        x_enc = self.transformer_encoder(x)
        out = x_enc[:, -1, :]
        return self.fc_out(out).squeeze(-1)

# -------------------------------
# Training utilities
# -------------------------------

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 30, lr: float = 1e-3, weight_decay: float = 1e-5,
                patience: int = 6, model_name: str = "model") -> Tuple[nn.Module, dict]:
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    history = {"train_loss": [], "val_loss": []}
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(DEVICE)
                yv = yv.to(DEVICE)
                vp = model(Xv)
                vloss = criterion(vp, yv)
                val_losses.append(vloss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.6f} val_loss: {val_loss:.6f}")

        if val_losses and val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("outputs", f"{model_name}.pt"))
    return model, history

# -------------------------------
# Rolling-origin CV with robust checks
# -------------------------------

def rolling_origin_evaluation(df_features: pd.DataFrame, feature_cols: List[str], target_col: str,
                              input_seq: int = 24, forecast_horizon: int = 1,
                              initial_train_size: int = 2000, step: int = 500,
                              model_type: str = 'lstm') -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    n = len(df_features)
    results = []
    preds_all = []
    trues_all = []

    for fold_start in range(initial_train_size, n - forecast_horizon, step):
        train_end = fold_start
        val_end = min(train_end + step, n - forecast_horizon)

        train_df = df_features.iloc[:train_end].copy()
        val_df = df_features.iloc[train_end:val_end].copy()

        # Check sizes
        if len(train_df) < input_seq + 1:
            print(f"Skipping fold at {fold_start}: train_df too small (len={len(train_df)})")
            continue
        if len(val_df) < input_seq + 1:
            print(f"Skipping fold at {fold_start}: val_df too small (len={len(val_df)})")
            continue

        def build_supervised(df_):
            Xs = []
            ys = []
            arr = df_[feature_cols].values
            yvals = df_[target_col].values
            for i in range(input_seq, len(df_) - forecast_horizon + 1):
                Xs.append(arr[i - input_seq:i, :])
                ys.append(yvals[i:i + forecast_horizon])
            if len(Xs) == 0:
                return np.empty((0,)), np.empty((0,))
            return np.stack(Xs), np.stack(ys)

        X_train, y_train = build_supervised(train_df)
        X_val, y_val = build_supervised(val_df)

        # If either is empty skip
        if X_train.size == 0 or X_val.size == 0:
            print(f"Skipping fold at {fold_start}: no supervised samples (X_train:{X_train.size}, X_val:{X_val.size})")
            continue

        # Scale features per-fold
        n_features = X_train.shape[2]
        scaler_X = StandardScaler()
        X_train_flat = X_train.reshape(-1, n_features)
        scaler_X.fit(X_train_flat)

        def apply_scale(X):
            b, s, f = X.shape
            return scaler_X.transform(X.reshape(-1, f)).reshape(b, s, f)

        X_train_s = apply_scale(X_train)
        X_val_s = apply_scale(X_val)

        scaler_y = StandardScaler()
        scaler_y.fit(y_train.reshape(-1, 1))
        y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
        y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).reshape(y_val.shape)

        train_ds = TimeSeriesDataset(X_train_s, y_train_s[:, -1])
        val_ds = TimeSeriesDataset(X_val_s, y_val_s[:, -1])

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        # instantiate
        if model_type == 'lstm':
            model = LSTMRegressor(input_size=n_features)
        else:
            model = TimeSeriesTransformer(input_size=n_features)

        model_name = f"{model_type}_fold_{fold_start}"
        model, history = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=6, model_name=model_name)

        # prepare test input: last input_seq rows ending at val_end - 1
        test_window = df_features.iloc[val_end - input_seq: val_end][feature_cols].values
        if len(test_window) < input_seq:
            print(f"Skipping prediction for fold {fold_start}: test_window too small (len={len(test_window)})")
            continue
        test_window_scaled = scaler_X.transform(test_window).reshape(1, input_seq, n_features)
        test_tensor = torch.tensor(test_window_scaled, dtype=torch.float32).to(DEVICE)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(test_tensor).cpu().numpy()
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)

        true_val = df_features.iloc[val_end][target_col]

        preds_all.append(pred)
        trues_all.append(np.array([true_val]))

        metrics = evaluate_metrics(np.array([true_val]), pred)
        results.append({"fold_start": fold_start, "metrics": metrics, "history": history})

    if len(preds_all) == 0:
        return results, np.array([]), np.array([])
    return results, np.concatenate(preds_all), np.concatenate(trues_all)

# -------------------------------
# Baseline
# -------------------------------

def baseline_exponential_smoothing(train_series: pd.Series, forecast_steps: int = 1):
    model = ExponentialSmoothing(train_series, seasonal_periods=24, trend='add', seasonal='add', damped_trend=True)
    fit = model.fit(optimized=True)
    return fit.forecast(forecast_steps)

# -------------------------------
# Explainability scaffolds
# -------------------------------

def explain_with_shap_lstm(model: nn.Module, X_sample: np.ndarray, feature_names: List[str], background: np.ndarray = None):
    if shap is None:
        raise RuntimeError("shap package is not installed; install with `pip install shap` to use explainability features")

    model_cpu = model.to('cpu')
    model_cpu.eval()

    # shap expects 2D input for KernelExplainer; we'll flatten sequences
    def predict_fn(x_flat):
        b = x_flat.shape[0]
        seq_len = X_sample.shape[1]
        n_features = X_sample.shape[2]
        x_reshaped = x_flat.reshape(b, seq_len, n_features).astype(np.float32)
        with torch.no_grad():
            t = torch.tensor(x_reshaped)
            out = model_cpu(t).numpy()
        return out

    if background is None:
        bg_idx = np.random.choice(len(X_sample), size=min(50, len(X_sample)), replace=False)
        background = X_sample[bg_idx]

    background_flat = background.reshape(len(background), -1)
    X_sample_flat = X_sample.reshape(len(X_sample), -1)

    explainer = shap.KernelExplainer(predict_fn, background_flat)
    shap_values = explainer.shap_values(X_sample_flat, nsamples=100)

    shap_values = np.array(shap_values).reshape(len(X_sample), X_sample.shape[1], X_sample.shape[2])
    return shap_values


def extract_attention_weights(transformer_model: TimeSeriesTransformer, X: np.ndarray) -> List[np.ndarray]:
    captured = []
    original_forwards = []

    for module in transformer_model.modules():
        if isinstance(module, nn.MultiheadAttention):
            original_forwards.append(module.forward)

            def make_forward(orig_forward):
                def forward(q, k, v, *args, **kwargs):
                    attn_output, attn_weights = orig_forward(q, k, v, *args, **kwargs)
                    try:
                        captured.append(attn_weights.detach().cpu().numpy())
                    except Exception:
                        captured.append(None)
                    return attn_output, attn_weights
                return forward

            module.forward = make_forward(module.forward)

    transformer_model.to(DEVICE)
    transformer_model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X.astype(np.float32)).to(DEVICE)
        _ = transformer_model(X_t)

    # restore
    i = 0
    for module in transformer_model.modules():
        if isinstance(module, nn.MultiheadAttention):
            module.forward = original_forwards[i]
            i += 1

    return captured

# -------------------------------
# Main workflow
# -------------------------------

def main():
    os.makedirs('outputs', exist_ok=True)

    print("Generating synthetic time series...")
    df = generate_complex_time_series(n_steps=3000)

    lags = [1, 24, 24 * 7]
    df_feat = create_features(df, lags=lags)

    feature_cols = [c for c in df_feat.columns if c not in ['ds', 'y']]
    target_col = 'y'

    print(f"Features: {feature_cols}")

    # Baseline quick check
    train_baseline = df_feat['y'].iloc[:2000]
    baseline_forecast = baseline_exponential_smoothing(train_baseline, forecast_steps=1)
    true_next = df_feat['y'].iloc[2000]
    baseline_metrics = evaluate_metrics(np.array([true_next]), baseline_forecast.values)
    print("Baseline metrics (Holt-Winters):", baseline_metrics)

    # Rolling-origin parameters
    input_seq = 48
    forecast_horizon = 1
    initial_train_size = 2000
    step = 400

    print("Running LSTM rolling-origin evaluation...")
    lstm_results, lstm_preds, lstm_trues = rolling_origin_evaluation(df_feat, feature_cols, target_col,
                                                                    input_seq=input_seq, forecast_horizon=forecast_horizon,
                                                                    initial_train_size=initial_train_size, step=step,
                                                                    model_type='lstm')
    if lstm_preds.size > 0:
        print("LSTM per-fold metrics:")
        for r in lstm_results:
            print(r['fold_start'], r['metrics'])

    print("Running Transformer rolling-origin evaluation...")
    trans_results, trans_preds, trans_trues = rolling_origin_evaluation(df_feat, feature_cols, target_col,
                                                                        input_seq=input_seq, forecast_horizon=forecast_horizon,
                                                                        initial_train_size=initial_train_size, step=step,
                                                                        model_type='transformer')

    def aggregate_metrics(list_of_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        if not list_of_metrics:
            return {"RMSE": float('nan'), "MAE": float('nan'), "MAPE": float('nan')}
        keys = list_of_metrics[0].keys()
        agg = {k: float(np.mean([m[k] for m in list_of_metrics])) for k in keys}
        return agg

    lstm_agg = aggregate_metrics([r['metrics'] for r in lstm_results])
    trans_agg = aggregate_metrics([r['metrics'] for r in trans_results])

    print("Aggregated metrics:")
    print("Baseline:", baseline_metrics)
    print("LSTM:", lstm_agg)
    print("Transformer:", trans_agg)

    summary = {"baseline": baseline_metrics, "lstm": lstm_agg, "transformer": trans_agg}
    with open('outputs/summary.pkl', 'wb') as f:
        pickle.dump(summary, f)

    # Visualize predictions if present
    plt.figure(figsize=(10, 4))
    if lstm_trues.size > 0:
        plt.plot(lstm_trues, label='True')
        plt.plot(lstm_preds, label='LSTM preds')
    if trans_trues.size > 0:
        plt.plot(trans_trues, label='True (Trans)', alpha=0.6)
        plt.plot(trans_preds, label='Transformer preds', alpha=0.8)
    plt.legend()
    plt.title('Rolling-origin single-step forecasts across folds')
    plt.savefig('outputs/rolling_preds.png', dpi=200)
    plt.close()

    # SHAP explainability demo (only if SHAP available and LSTM model saved)
    if shap is not None:
        print("Attempting SHAP explainability demo (LSTM)...")
        # Build sample sequences
        seq_len = input_seq
        arr = df_feat[feature_cols].values
        X_all = []
        for i in range(seq_len, len(df_feat)):
            X_all.append(arr[i - seq_len:i, :])
        X_all = np.stack(X_all)

        lstm_model_path = os.path.join('outputs', f'lstm_fold_{initial_train_size}.pt')
        if os.path.exists(lstm_model_path):
            lstm_model = LSTMRegressor(input_size=len(feature_cols))
            lstm_model.load_state_dict(torch.load(lstm_model_path, map_location='cpu'))
            sample_idx = np.random.choice(len(X_all), size=min(30, len(X_all)), replace=False)
            X_sample = X_all[sample_idx]
            try:
                shap_vals = explain_with_shap_lstm(lstm_model, X_sample, feature_cols)
                np.save('outputs/shap_values_lstm.npy', shap_vals)
                print('Saved SHAP values to outputs/shap_values_lstm.npy')
            except Exception as e:
                print('SHAP explainability failed:', e)
        else:
            print('No LSTM model file found for SHAP demo.')
    else:
        print('SHAP not installed; skipping explainability demo.')

    print('Done. Outputs saved in ./outputs/')

if __name__ == '__main__':
    main()

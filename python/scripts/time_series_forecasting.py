import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

from tora.client import Client as Essistant


def safe_value(value):
    """Convert any value to float for logging, return None for strings"""
    if isinstance(value, (int, float)):
        # Handle special float values
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    elif isinstance(value, bool):
        return int(value)
    elif isinstance(value, str):
        return None  # Skip string values
    else:
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


def log_metric(client, name, value, step):
    """Log only numeric metrics"""
    value = safe_value(value)
    if value is not None:
        client.log(name=name, value=value, step=step)


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


def create_sequences(data, seq_length, horizon):
    xs, ys = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length:i+seq_length+horizon])
    return np.array(xs), np.array(ys)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, essistant):
    model.train()
    running_loss = 0.0
    total_samples = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        try:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            if batch_idx % 20 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
    
    # Calculate metrics
    epoch_loss = running_loss / max(total_samples, 1)
    epoch_time = time.time() - start_time
    
    # Log metrics
    log_metric(essistant, "train_loss", epoch_loss, epoch)
    log_metric(essistant, "epoch_time", epoch_time, epoch)
    
    return epoch_loss


def validate(model, device, val_loader, criterion, scaler, epoch, essistant, split="val"):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            # Convert predictions back to original scale for metric calculation
            # Assuming the last feature is the target
            scaled_preds = output.cpu().numpy()
            scaled_targets = target.cpu().numpy()
            
            # For sequence-to-sequence models, we might need to reshape
            if len(scaled_preds.shape) > 2:
                scaled_preds = scaled_preds.reshape(-1, scaled_preds.shape[-1])
                scaled_targets = scaled_targets.reshape(-1, scaled_targets.shape[-1])
            
            # Inverse transform if scaler is provided
            if scaler:
                # Create dummy array matching scaler's expected shape
                dummy = np.zeros((scaled_preds.shape[0], scaler.scale_.shape[0]))
                # Place predictions in the last column (assuming univariate forecasting)
                dummy[:, -1] = scaled_preds[:, 0]  
                # Inverse transform
                preds_orig = scaler.inverse_transform(dummy)[:, -1]
                
                # Same for targets
                dummy = np.zeros((scaled_targets.shape[0], scaler.scale_.shape[0]))
                dummy[:, -1] = scaled_targets[:, 0]
                targets_orig = scaler.inverse_transform(dummy)[:, -1]
            else:
                preds_orig = scaled_preds[:, 0]
                targets_orig = scaled_targets[:, 0]
            
            all_predictions.extend(preds_orig)
            all_targets.extend(targets_orig)
    
    # Calculate validation loss
    val_loss = running_loss / max(total_samples, 1)
    
    # Calculate regression metrics
    try:
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
    except:
        mse, rmse, mae, r2 = 0, 0, 0, 0
    
    # Log metrics
    prefix = "val" if split == "val" else "test"
    log_metric(essistant, f"{prefix}_loss", val_loss, epoch)
    log_metric(essistant, f"{prefix}_mse", mse, epoch)
    log_metric(essistant, f"{prefix}_rmse", rmse, epoch)
    log_metric(essistant, f"{prefix}_mae", mae, epoch)
    log_metric(essistant, f"{prefix}_r2", r2, epoch)
    
    print(
        f"\n{split.capitalize()} set: Loss: {val_loss:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}\n"
    )
    
    return val_loss, mse, rmse, mae, r2


def generate_synthetic_data():
    """Generate synthetic time series data with trend, seasonality and noise"""
    np.random.seed(42)
    time_steps = 1000
    
    # Time index
    t = np.arange(time_steps)
    
    # Trend component
    trend = 0.01 * t
    
    # Seasonal component with multiple frequencies
    season1 = 2 * np.sin(2 * np.pi * t / 50)  # Period of 50 time steps
    season2 = 1 * np.sin(2 * np.pi * t / 100)  # Period of 100 time steps
    seasonality = season1 + season2
    
    # Noise component
    noise = 0.5 * np.random.randn(time_steps)
    
    # Combine components
    data = trend + seasonality + noise
    
    # Add features that might be useful
    features = np.column_stack([
        data,  # The target series itself
        np.sin(2 * np.pi * t / 50),  # Sine feature with period 50
        np.cos(2 * np.pi * t / 50),  # Cosine feature with period 50
        np.sin(2 * np.pi * t / 100),  # Sine feature with period 100
        np.cos(2 * np.pi * t / 100),  # Cosine feature with period 100
        t / 1000.0  # Normalized time index for trend
    ])
    
    df = pd.DataFrame(
        features,
        columns=['value', 'sin_50', 'cos_50', 'sin_100', 'cos_100', 'time']
    )
    
    return df


if __name__ == "__main__":
    # Hyperparameters
    hyperparams = {
        "batch_size": 32,
        "epochs": 50,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout_rate": 0.2,
        "seq_length": 24,  # Look back window
        "horizon": 12,     # Forecast horizon
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "patience": 5,
        "factor": 0.5,
        "min_lr": 1e-6,
        "early_stopping": 10,
    }
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyperparams["device"] = str(device)
    
    # Load or generate dataset
    try:
        # Try to load real world data (you can replace this with your own dataset)
        df = pd.read_csv('data/time_series_data.csv')
        print("Loaded real world dataset")
    except:
        # Generate synthetic data if no real data is available
        print("Generating synthetic time series data")
        df = generate_synthetic_data()
    
    # Prepare data
    data = df.values  # Convert to numpy array
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(
        data_scaled, 
        seq_length=hyperparams["seq_length"], 
        horizon=hyperparams["horizon"]
    )
    
    # Create dataset
    dataset = TimeSeriesDataset(X, y)
    
    # Split into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=hyperparams["batch_size"], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=hyperparams["batch_size"]
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=hyperparams["batch_size"]
    )
    
    # Update hyperparams with dataset info
    hyperparams.update({
        "input_size": X.shape[2],  # Number of features
        "output_size": y.shape[1],  # Forecast horizon
        "train_samples": train_size,
        "val_samples": val_size,
        "test_samples": test_size,
    })
    
    # Initialize the model
    model = LSTMForecaster(
        input_size=hyperparams["input_size"],
        hidden_size=hyperparams["hidden_size"],
        num_layers=hyperparams["num_layers"],
        output_size=hyperparams["output_size"],
        dropout_rate=hyperparams["dropout_rate"]
    ).to(device)
    
    # Count parameters
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hyperparams["model_parameters"] = model_params
    
    # Initialize experiment tracker
    essistant = Essistant(
        name="TimeSeries_LSTM",
        description="LSTM model for time series forecasting with tracked metrics",
        hyperparams=hyperparams,
        tags=["time-series", "forecasting", "lstm", "regression"],
    )
    
    # Initialize loss function and optimizer
    criterion = nn.MSELoss()
    
    if hyperparams["optimizer"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"]
        )
    else:
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"]
        )
    
    # Initialize learning rate scheduler
    if hyperparams["scheduler"] == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=hyperparams["factor"],
            patience=hyperparams["patience"],
            min_lr=hyperparams["min_lr"],
            verbose=True
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=10, 
            gamma=0.5
        )
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = "best_time_series_model.pt"
    early_stopping_counter = 0
    
    for epoch in range(1, hyperparams["epochs"] + 1):
        # Log current learning rate
        log_metric(essistant, "learning_rate", optimizer.param_groups[0]["lr"], epoch)
        
        # Train for one epoch
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion, epoch, essistant)
        
        # Validate
        val_loss, val_mse, val_rmse, val_mae, val_r2 = validate(
            model, device, val_loader, criterion, scaler, epoch, essistant, split="val"
        )
        
        # Update learning rate based on validation loss
        if hyperparams["scheduler"] == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation loss: {best_val_loss:.6f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        # Early stopping
        if early_stopping_counter >= hyperparams["early_stopping"]:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Evaluate best model on test set
    print(f"Loading best model with validation loss: {best_val_loss:.6f}")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_mse, test_rmse, test_mae, test_r2 = validate(
        model, device, test_loader, criterion, scaler, hyperparams["epochs"], essistant, split="test"
    )
    
    # Log final metrics
    log_metric(essistant, "final_test_loss", test_loss, hyperparams["epochs"])
    log_metric(essistant, "final_test_mse", test_mse, hyperparams["epochs"])
    log_metric(essistant, "final_test_rmse", test_rmse, hyperparams["epochs"])
    log_metric(essistant, "final_test_mae", test_mae, hyperparams["epochs"])
    log_metric(essistant, "final_test_r2", test_r2, hyperparams["epochs"])
    
    # Generate forecasts for a sample from the test set
    model.eval()
    with torch.no_grad():
        sample_data, sample_target = next(iter(test_loader))
        sample_data = sample_data.to(device)
        sample_forecast = model(sample_data)
        
        # Convert back to original scale
        sample_forecast = sample_forecast.cpu().numpy()
        sample_target = sample_target.cpu().numpy()
        
        # For visualization - just take the first batch item
        forecast = sample_forecast[0]
        actual = sample_target[0]
        
        # Inverse transform if using a scaler
        if scaler:
            # Create dummy arrays for inverse transform
            dummy_forecast = np.zeros((forecast.shape[0], scaler.scale_.shape[0]))
            dummy_actual = np.zeros((actual.shape[0], scaler.scale_.shape[0]))
            
            # Place values in the target column
            dummy_forecast[:, -1] = forecast
            dummy_actual[:, -1] = actual
            
            # Inverse transform
            forecast = scaler.inverse_transform(dummy_forecast)[:, -1]
            actual = scaler.inverse_transform(dummy_actual)[:, -1]
    
    # Create a save directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(actual)), actual, label='Actual')
    plt.plot(range(len(forecast)), forecast, label='Forecast', linestyle='--')
    plt.title('Time Series Forecast vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/time_series_forecast.png')
    
    # Clean up
    essistant.shutdown()
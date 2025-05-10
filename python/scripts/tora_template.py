#!/usr/bin/env python3
"""
Tora Training Template

This template provides a standardized structure for creating new training scripts with
Tora integration for experiment tracking. Customize the sections as needed for your specific
model and dataset.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tora import Tora
from torch.utils.data import DataLoader

# Additional imports based on your specific needs
# from torchvision import datasets, transforms, models
# from transformers import AutoModelForSequenceClassification, AutoTokenizer


def safe_value(value):
    """Convert values to a format suitable for Tora logging, handling edge cases."""
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    elif isinstance(value, bool):
        return int(value)
    elif isinstance(value, str):
        return None
    else:
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


def log_metric(client, name, value, step):
    """Log a metric to Tora after ensuring it's a valid value."""
    value = safe_value(value)
    if value is not None:
        client.log(name=name, value=value, step=step)


# ===== MODEL DEFINITION =====
class YourModel(nn.Module):
    """Define your model architecture here."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(YourModel, self).__init__()
        # Define model layers
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Define forward pass
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


# ===== DATASET HANDLING =====
def load_dataset():
    """
    Load and prepare your dataset.
    Returns train, validation, and test datasets.
    """
    # Example implementation (modify as needed):
    # train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    # train_size = int(0.8 * len(train_dataset))
    # val_size = len(train_dataset) - train_size
    # train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    # test_dataset = datasets.MNIST("data", train=False, transform=transform)

    # Replace with your dataset loading logic
    train_set = None
    val_set = None
    test_dataset = None

    return train_set, val_set, test_dataset


# ===== TRAINING FUNCTIONS =====
def train_epoch(model, device, train_loader, optimizer, criterion, epoch, tora):
    """Train for one epoch and log metrics to Tora."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        try:
            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Print progress
            if batch_idx % 50 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")

    # Calculate and log epoch metrics
    epoch_loss = running_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    epoch_time = time.time() - start_time

    log_metric(tora, "train_loss", epoch_loss, epoch)
    log_metric(tora, "train_accuracy", accuracy, epoch)
    log_metric(tora, "epoch_time", epoch_time, epoch)

    return epoch_loss, accuracy


def validate(model, device, test_loader, criterion, epoch, tora, split="val"):
    """Evaluate model on validation/test data and log metrics to Tora."""
    model.eval()
    test_loss = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())

    dataset_size = len(test_loader.dataset)
    test_loss = test_loss / max(dataset_size, 1)

    try:
        accuracy = accuracy_score(all_targets, all_predictions) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average="weighted", zero_division=0
        )
    except:
        accuracy, precision, recall, f1 = 0, 0, 0, 0

    # Log metrics with appropriate prefix
    prefix = "val" if split == "val" else "test"
    log_metric(tora, f"{prefix}_loss", test_loss, epoch)
    log_metric(tora, f"{prefix}_accuracy", accuracy, epoch)
    log_metric(tora, f"{prefix}_precision", precision * 100, epoch)
    log_metric(tora, f"{prefix}_recall", recall * 100, epoch)
    log_metric(tora, f"{prefix}_f1", f1 * 100, epoch)

    print(
        f"\n{split.capitalize()} set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1 * 100:.2f}%\n"
    )

    return test_loss, accuracy, precision, recall, f1


def log_per_class_metrics(model, device, data_loader, class_names, tora, epoch):
    """Log detailed per-class metrics to Tora."""
    all_targets = []
    all_predictions = []
    model.eval()

    # Collect predictions
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(pred.cpu().numpy())

    try:
        cm = confusion_matrix(all_targets, all_predictions)
        num_classes = len(class_names) if class_names else cm.shape[0]

        # Calculate metrics for each class
        for class_idx in range(num_classes):
            true_positives = cm[class_idx, class_idx]
            false_positives = cm[:, class_idx].sum() - true_positives
            false_negatives = cm[class_idx, :].sum() - true_positives

            denominator_p = max(true_positives + false_positives, 1)
            denominator_r = max(true_positives + false_negatives, 1)

            class_precision = true_positives / denominator_p
            class_recall = true_positives / denominator_r

            if class_precision + class_recall > 0:
                class_f1 = (
                    2
                    * (class_precision * class_recall)
                    / (class_precision + class_recall)
                )
            else:
                class_f1 = 0

            # Get class name (if available) or use index
            class_name = class_names[class_idx] if class_names else str(class_idx)

            # Log metrics
            log_metric(
                tora, f"class_{class_name}_precision", class_precision * 100, epoch
            )
            log_metric(tora, f"class_{class_name}_recall", class_recall * 100, epoch)
            log_metric(tora, f"class_{class_name}_f1", class_f1 * 100, epoch)
    except Exception as e:
        print(f"Error calculating per-class metrics: {str(e)}")


# ===== MAIN FUNCTION =====
def main():
    # ===== HYPERPARAMETERS =====
    hyperparams = {
        # Basic training parameters
        "batch_size": 32,
        "epochs": 10,
        "lr": 0.001,
        "weight_decay": 5e-4,
        # Model architecture parameters
        "input_dim": 784,  # Example for MNIST
        "hidden_dim": 128,
        "output_dim": 10,  # Number of classes
        "dropout_rate": 0.2,
        # Optimizer parameters
        "optimizer": "SGD",  # "SGD" or "Adam"
        "momentum": 0.9,
        "nesterov": True,
        "dampening": 0,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        # Scheduler parameters
        "scheduler": "cosine",  # "cosine", "linear", "step", etc.
        "step_size": 5,  # For StepLR
        "gamma": 0.1,  # For StepLR
    }

    # ===== DEVICE SETUP =====
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    hyperparams["device"] = str(device)

    # ===== DATA LOADING =====
    train_set, val_set, test_dataset = load_dataset()

    # Create data loaders
    train_loader = DataLoader(
        train_set, batch_size=hyperparams["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_set, batch_size=hyperparams["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"])

    # ===== MODEL CREATION =====
    model = YourModel(
        input_dim=hyperparams["input_dim"],
        hidden_dim=hyperparams["hidden_dim"],
        output_dim=hyperparams["output_dim"],
        dropout_rate=hyperparams["dropout_rate"],
    ).to(device)

    # Count model parameters
    model_params = sum(p.numel() for p in model.parameters())

    # Update hyperparams with model info
    hyperparams.update(
        {
            "model": "YourModel",
            "model_parameters": model_params,
            "dataset": "YourDataset",
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "test_samples": len(test_dataset),
            "criterion": "CrossEntropyLoss",
        }
    )

    # ===== TORA EXPERIMENT SETUP =====
    tora = Tora.create_experiment(
        name="Your_Experiment_Name",
        description="Description of your model and experiment",
        hyperparams=hyperparams,
        tags=["tag1", "tag2", "tag3"],  # Replace with relevant tags
    )

    # ===== TRAINING SETUP =====
    criterion = nn.CrossEntropyLoss()

    # Configure optimizer based on hyperparameters
    if hyperparams["optimizer"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparams["lr"],
            momentum=hyperparams["momentum"],
            weight_decay=hyperparams["weight_decay"],
            nesterov=hyperparams["nesterov"],
            dampening=hyperparams["dampening"],
        )
    elif hyperparams["optimizer"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
            betas=(hyperparams["beta1"], hyperparams["beta2"]),
            eps=hyperparams["eps"],
        )
    else:
        optimizer = optim.SGD(model.parameters(), lr=hyperparams["lr"])

    # Configure learning rate scheduler
    if hyperparams["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hyperparams["epochs"]
        )
    elif hyperparams["scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=hyperparams["step_size"], gamma=hyperparams["gamma"]
        )
    else:
        scheduler = None

    # Variables to track best model
    best_val_acc = 0
    best_model_path = "best_model.pt"

    # ===== TRAINING LOOP =====
    for epoch in range(1, hyperparams["epochs"] + 1):
        # Log learning rate
        log_metric(tora, "learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # Train
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch, tora
        )

        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, device, val_loader, criterion, epoch, tora, split="val"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation accuracy: {best_val_acc:.2f}%")

        # Step scheduler if defined
        if scheduler:
            scheduler.step()

    # ===== FINAL EVALUATION =====
    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))

    # Evaluate on test set
    test_loss, test_acc, test_prec, test_rec, test_f1 = validate(
        model, device, test_loader, criterion, hyperparams["epochs"], tora, split="test"
    )

    # Log final metrics
    log_metric(tora, "final_test_accuracy", test_acc, hyperparams["epochs"])
    log_metric(tora, "final_test_precision", test_prec * 100, hyperparams["epochs"])
    log_metric(tora, "final_test_recall", test_rec * 100, hyperparams["epochs"])
    log_metric(tora, "final_test_f1", test_f1 * 100, hyperparams["epochs"])

    # Log per-class metrics
    class_names = ["class1", "class2", "class3"]  # Replace with your class names
    log_per_class_metrics(
        model, device, test_loader, class_names, tora, hyperparams["epochs"]
    )

    # Shutdown Tora client
    tora.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

# Standard library imports
from numbers import Number
from pathlib import Path
from typing import Dict, List, Tuple

# Third party imports
import numpy as np
import pandas as pd
import torch
from torch.nn.modules.loss import MSELoss
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# Local imports
from mlp.mlp import MLP


def train_test_ae_mlp(
    model: MLP,
    tr_loader: DataLoader, val_loader: DataLoader, te_loader: DataLoader,
    criterion: MSELoss, optimizer: Adam, scheduler: StepLR, 
    lr: float, step_size: int, epochs: int,
    models_dir: Path, scores_dir: Path,
    device: torch.device
    ) -> Tuple[List[Dict[str, Number]], List[Dict[str, Number]]]:
    """
    Train an autoencoder multilayer perceptron.
    Arguments:
        model: MLP model.
        tr_loader: Training data loader.
        val_loader: Validation data loader.
        te_loader: Test data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        lr: Learning rate.
        step_size: Step size for learning rate scheduler.
        epochs: Number of epochs.
        models_dir: Directory to save models.
        scores_dir: Directory to save scores.
        device: Device to train on.
    """
    tr_val_losses = []
    te_losses = []
    min_val_loss = np.inf
    # Train model and get train loss
    for epoch in range(1, epochs + 1):
        tr_loss = 0
        for _, batch in enumerate(tr_loader):
            batch = batch.to(device)
            # Forward
            model.train()
            output = model(batch)
            loss = criterion(output, batch)
            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Increment training loss
            tr_loss += loss.item()
        tr_loss_per_batch = tr_loss / len(tr_loader)
        print(f"[{step_size}][{lr:.3f}][{epoch}]: tr_loss_per_batch={tr_loss_per_batch:.4f}")
        
        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _, val_batch in enumerate(val_loader):
                val_batch = val_batch.to(device)
                output = model(val_batch)
                loss = criterion(output, val_batch)
                val_loss += loss.item()
        val_loss_per_batch = val_loss / len(val_loader)
        print(f"[{step_size}][{lr:.3f}][{epoch}]: val_loss_per_batch={val_loss_per_batch:.4f}")

        print(f"[{step_size}][{lr:.3f}][{epoch}]: batch_val_loss={val_loss_per_batch:.4f}, lr={lr:.6f}")
        if val_loss_per_batch < min_val_loss:
            min_val_loss = val_loss_per_batch
            torch.save(model.state_dict(), models_dir / f"mlp-autoencoder-{step_size}-{lr:.3f}.pt")
            print(f"[{step_size}][{lr:.3f}][{epoch}]: Saved model b/c val_loss at all-time low.")
        else:
            print(f"[{step_size}][{lr:.3f}][{epoch}]: Did not save model b/c val_loss too high.")
        
        # Organize losses
        tr_val_losses.append(dict(
            step_size=step_size,
            lr=lr,
            stepped_lr=scheduler.get_last_lr()[0],
            epoch=epoch,
            tr_loss_per_batch=tr_loss_per_batch,
            val_loss_per_batch=val_loss_per_batch,
        ))
        
        # Decrease learning rate
        scheduler.step()
        
    # Test loss
    te_loss = 0
    with torch.no_grad():
        for _, te_batch in enumerate(te_loader):
            te_batch = te_batch.to(device)
            output = model(te_batch)
            loss = criterion(output, te_batch)
            te_loss += loss.item()
    te_loss_per_batch = te_loss / len(te_loader)
    te_losses.append(dict(
        step_size=step_size,
        lr=lr,
        stepped_lr=scheduler.get_last_lr()[0],
        te_loss_per_batch=te_loss_per_batch,
    ))
    
    # Save losses
    (
        pd.DataFrame(tr_val_losses)
        .set_index(["step_size", "lr", "epoch"])
        .to_csv(scores_dir / "te_losses.csv")
    )
    (
        pd.DataFrame(te_losses)
        .set_index(["step_size", "lr"])
        .to_csv(scores_dir / "te_losses.csv")
    )
    
    return tr_val_losses, te_losses
# Standard library imports
from numbers import Number
from pathlib import Path
from typing import Dict, List, Tuple

# Third party imports
import numpy as np
import pandas as pd
import rasterio as rio
import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    """Dataset of image patches."""
    def __init__(self, images_dir: Path, output_bands: List[str], slice_frac: float=0.001):
        self.images_dir = images_dir
        self.output_bands = output_bands
        self.slice_frac = slice_frac
        
        self.cogs_meta = pd.Series(images_dir.glob("*.tif"), name="path").to_frame()
        self.cogs_meta["filestem"] = self.cogs_meta["path"].apply(lambda x: x.stem)
        self.cogs_meta["event_key"] = self.cogs_meta["filestem"].apply(lambda x: x.split("_")[0])
        self.cogs_meta["solar_date"] = self.cogs_meta["filestem"].apply(lambda x: x.split("_")[1])
        
        cube = []
        for scene_src in self.cogs_meta["path"].values:
            with rio.open(scene_src) as src_scene:
                arr = src_scene.read()
                bands, rows, cols = arr.shape
                arr = np.moveaxis(arr, 0, -1)
                arr = arr.reshape(-1, bands)
                slice_indices = (np.random.rand(int(slice_frac * rows * cols)) * rows * cols).astype(int)
                cube.append(arr[slice_indices])
        self.cube = np.concatenate(cube, axis=0).astype(float)
        for band_ix, _ in enumerate(output_bands):
            min_ = self.cube[:, band_ix].mean()
            std = self.cube[:, band_ix].std()
            self.cube[:, band_ix] = (self.cube[:, band_ix] - min_) / std    
        self.cube = self.cube.astype(np.float32)
        del cube
        
    def __len__(self) -> int:
        return len(self.cube)

    def __getitem__(self, index) -> torch.Tensor:
        return torch.from_numpy(self.cube[index])
    

class MLP(nn.Module):
    """Multilayer perceptron model."""
    def __init__(self, in_features: int, hidden_neurons: int):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=in_features, out_features=hidden_neurons
        )
        self.decoder_output_layer = nn.Linear(
            in_features=hidden_neurons, out_features=in_features
        )

    def forward(self, features):
        hidden_z = self.encoder_hidden_layer(features)
        hidden_y = torch.relu(hidden_z)
        output_z = self.decoder_output_layer(hidden_y)
        output_y = torch.relu(output_z)
        return output_y


def train_test_ae_mlp(
    model,
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
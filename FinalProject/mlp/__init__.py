#!/usr/bin/env python3

# Third party imports
from torch.utils.data import DataLoader

# Local imports
from mlp.datasets import MLPDataset
from mlp.mlp import MLP
from mlp.train import train_test_ae_mlp


def encode(loader: DataLoader, model: MLP, scenes: int, rows: int, cols: int) -> NDArray:
    """
    Encode scenes in loader with model.
    Arguments:
        loader: DataLoader of scenes.
        model: MLP model.
        scenes: Number of scenes.
        rows: Number of rows.
        cols: Number of columns.
    Returns: Encoded scenes.
    """
    out = []
    with torch.inference_mode():
        for scene in loader:
            scene = scene.to(device)
            out.append(model.encoder_hidden_layer(scene).detach().cpu().numpy())
    return np.concatenate(out, axis=0).reshape(scenes, rows, cols, 2)
from rich.progress import track
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_fn(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, device: torch.device) -> float:
    """
    Train the model on the provided data using the given optimizer.

    Parameters:
        model (nn.Module): A PyTorch model to be trained.
        data_loader (DataLoader): A PyTorch DataLoader object that provides batches of training data.
        optimizer (Optimizer): A PyTorch optimizer used to update the model's parameters.
        device (torch.device): A PyTorch device (such as 'cpu' or 'cuda') where the data and model should be loaded.

    Returns:
        fin_loss (float): The average loss across all batches of training data.
    """
    model.train()
    fin_loss = 0

    for data in track(data_loader, description="😪 Training..."):
        for key, value in data.items():
            data[key] = value.to(device)

        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_loader)


def eval_fn(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[List[Any], float]:
    """
    Evaluate the model on the provided data.

    Parameters:
        model (nn.Module): A PyTorch model to be evaluated.
        data_loader (DataLoader): A PyTorch DataLoader object that provides batches of evaluation data.
        device (torch.device): A PyTorch device (such as 'cpu' or 'cuda') where the data and model should be loaded.

    Returns:
        fin_preds (list): A list of predictions made by the model on the evaluation data.
        fin_loss (float): The average loss across all batches of evaluation data.
    """
    model.eval()
    with torch.no_grad():
        fin_loss = 0
        fin_preds = []
        for data in track(data_loader, description="🤔 Testing ..."):
            for key, value in data.items():
                data[key] = value.to(device)

            batch_preds, loss = model(**data)
            fin_loss += loss.item()
            fin_preds.append(batch_preds)
        return fin_preds, fin_loss / len(data_loader)

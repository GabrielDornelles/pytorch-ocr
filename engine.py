from tqdm import tqdm
import torch
import config
import time
from rich.progress import track

def train_fn(model, data_loader, optimizer):
    model.train()
    fin_loss = 0

    for data in track(data_loader, description="😪 Training..."):
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)

        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_loader)

def eval_fn(model, data_loader):
    model.eval()
    with torch.no_grad(): # model forward steps is requires_grad=True by default, inference will never need to calculate gradients so disable it and save memory
        fin_loss = 0
        fin_preds = []
        for data in track(data_loader,description="🤔 Testing ..."):
            for key, value in data.items():
                data[key] = value.to(config.DEVICE)

            batch_preds, loss = model(**data)
            fin_loss += loss.item()
            fin_preds.append(batch_preds)
        return fin_preds, fin_loss / len(data_loader)
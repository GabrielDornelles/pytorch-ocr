import torch
from rich.progress import track

def train_fn(model, data_loader, optimizer, device):
    model.train()
    fin_loss = 0

    for data in track(data_loader, description="😪 Training..."):
        for key, value in data.items():
            data[key] = value.to(device)

        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_loader)

def eval_fn(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        fin_loss = 0
        fin_preds = []
        for data in track(data_loader,description="🤔 Testing ..."):
            for key, value in data.items():
                data[key] = value.to(device)

            batch_preds, loss = model(**data)
            fin_loss += loss.item()
            fin_preds.append(batch_preds)
        return fin_preds, fin_loss / len(data_loader)
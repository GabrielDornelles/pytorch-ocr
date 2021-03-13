from tqdm import tqdm
import torch
import config
import time

#TODO: FIX: RuntimeError: stack expects each tensor to be equal size, but got [6] at entry 0 and [8] at entry 11 
# tensors can't handle data with different length of characters, use image_validation.py.
def train_fn(model, data_loader, optimizer):
    model.train()
    fin_loss = 0
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for data in tk0:
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
        tk0 = tqdm(data_loader, total=len(data_loader))
        for data in tk0:
            for key, value in data.items():
                data[key] = value.to(config.DEVICE)
            batch_preds, loss = model(**data)
            fin_loss += loss.item()
            fin_preds.append(batch_preds)
        #print(f"batch preds: {batch_preds}") #print batch tensors of predictions
        return fin_preds, fin_loss / len(data_loader)
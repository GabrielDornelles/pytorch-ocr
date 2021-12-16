import os
import glob
from datetime import datetime
from collections import Counter

import torch
import numpy as np
from torch import nn
import copy

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
import engine
from model import OcrModel
from plot import plot_acc, plot_losses

from rich.console import Console
from rich.table import Table

console = Console()
torch.cuda.empty_cache()

classes = ['∅', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def decode_predictions(predictions):
    """
    CTC RNN Layer decoder.
    1.
        2 (or more) repeating digits are collapsed into a single instance of that digit unless 
        separated by blank - this compensates for the fact that the RNN performs a classification 
        for each stripe that represents a part of a digit (thus producing duplicates)
    2.
        Multiple consecutive blanks are collapsed into one blank - this compensates 
        for the spacing before, after or between the digits
    """
    predictions = predictions.permute(1, 0, 2)
    predictions = torch.softmax(predictions, 2)
    predictions = torch.argmax(predictions, 2)
    predictions = predictions.detach().cpu().numpy()
    texts = []
    for i in range(predictions.shape[0]):
        string = ""
        batch_e = predictions[i]
        
        for j in range(len(batch_e)):
            string += classes[batch_e[j]]

        string = string.split("∅")
        string = [x for x in string if x != ""]
        string = [list(set(x))[0] for x in string]
        texts.append(''.join(string))
    return texts

def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.png"))
    targets_orig = [x.split("/")[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    targets_enc = [lbl_enc.transform(x) for x in targets]
    targets_enc = np.array(targets_enc)
    targets_enc = targets_enc + 1

    (
        train_imgs,
        test_imgs,
        train_targets,
        test_targets,
        _,
        test_targets_orig,
    ) = model_selection.train_test_split(
        image_files, targets_enc, targets_orig, test_size=0.1, random_state=42
    )

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )
    test_dataset = dataset.ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    print(f"num of classes: {len(lbl_enc.classes_)}")
    print(f"classes: {lbl_enc.classes_}")
    model = OcrModel(num_chars=len(lbl_enc.classes_))
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    best_acc = 0.0
    train_loss_data = []
    valid_loss_data = []
    accuracy_data = []
    start=datetime.now()

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        train_loss_data.append(train_loss)
        valid_preds, test_loss = engine.eval_fn(model, test_loader)
        valid_loss_data.append(test_loss)
        valid_captcha_preds = []
        for vp in valid_preds:
            current_preds = decode_predictions(vp)
            valid_captcha_preds.extend(current_preds)

        combined = list(zip(test_targets_orig, valid_captcha_preds))
        if config.VIEW_INFERENCE_WHILE_TRAINING:
            table = Table(show_header=True, header_style="hot_pink")
            table.add_column("Ground Truth", width=12)
            table.add_column("Predicted")
            table.border_style = "bright_yellow"
            table.columns[0].style = "violet"
            table.columns[1].style = "grey93"
            for idx in combined[:]:
                table.add_row(idx[0]
                ,idx[1])
            console.print(table)
           
        test_dup_rem = test_targets_orig
        accuracy = metrics.accuracy_score(test_dup_rem, valid_captcha_preds)
    
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Epoch", style="aquamarine1", width=12)
        table.add_column("Train Loss", style="bright_green")
        table.add_column("Test Loss", style="bright_green")
        table.add_column("Accuracy", style="bright_yellow")
        table.add_column("Best Accuracy", style="gold1")
        table.columns[0].header_style = "aquamarine1"
        table.columns[1].header_style = "bright_green"
        table.columns[2].header_style = "bright_green"
        table.columns[3].header_style = "bright_yellow"
        table.columns[4].header_style = "bright_yellow"
        
        accuracy_data.append(accuracy)
        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            if config.SAVE_CHECKPOINTS:
                torch.save(model, f"checkpoint-{(best_acc*100):.2f}.pth")

        scheduler.step(test_loss)
        table.add_row(str(epoch),
        str(train_loss),
        str(test_loss),
        str(accuracy), 
        str(best_acc))
        console.print(table)

    print(f"final best accuracy: {(best_acc*100):.2f}%")

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), config.SAVE_MODEL_AS)

    print(f"Saving model on {config.SAVE_MODEL_AS}\nTraining time: {datetime.now()-start}")
    plot_losses(train_loss_data, valid_loss_data)
    plot_acc(accuracy_data)
   

if __name__ == "__main__":
    try:
        run_training()
    except Exception:
        console.print_exception()

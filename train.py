import os
import glob
from datetime import datetime

import torch
from torch import nn
import numpy as np
import copy
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import config
import dataset
import engine
from model import OcrModel
from plot import plot_acc, plot_losses

torch.cuda.empty_cache()

def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin

def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))
    return cap_preds


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
            current_preds = decode_predictions(vp, lbl_enc)
            valid_captcha_preds.extend(current_preds)

        combined = list(zip(test_targets_orig, valid_captcha_preds))
        if config.VIEW_INFERENCE_WHILE_TRAINING:
            print(f"validations: {combined}") # combined[:10] print right answer vs predicted answer, first 10 from batch
        test_dup_rem = [remove_duplicates(c) for c in test_targets_orig]
        accuracy = metrics.accuracy_score(test_dup_rem, valid_captcha_preds)
        print(
            f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={test_loss} Accuracy={accuracy}"
        )
        accuracy_data.append(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            if config.SAVE_CHECKPOINTS:
                torch.save(model, f"checkpoint-{(best_acc*100):.2f}.pth")

        scheduler.step(test_loss)
        print(f"best accuracy: {(best_acc*100):.2f}%")
    print(f"final best accuracy: {(best_acc*100):.2f}%")

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), config.SAVE_MODEL_AS)

    print(f"Saving model on {config.SAVE_MODEL_AS}\nTraining time: {datetime.now()-start}")
    plot_losses(train_loss_data, valid_loss_data)
    plot_acc(accuracy_data)
   

if __name__ == "__main__":
    run_training()
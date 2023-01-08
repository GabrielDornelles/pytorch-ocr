import os
import glob
import copy
import hydra
from omegaconf import OmegaConf
from datetime import datetime

import torch
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

from rich.console import Console
from rich.table import Table
from rich import print
import logging

import dataset
import engine

from models.crnn import CRNN 
from utils.plot import plot_acc, plot_losses
from utils.model_decoders import decode_predictions, decode_padded_predictions


# Setup rich console
console = Console()

def setup_logging():
    # This function overwrites 'datefmt' from hydra default logger
    # TODO: is there anyway to set datefmt direclty at configs/hydra/job_logging/custom.yaml?
    logger = logging.getLogger()
    formatter = logging.Formatter(fmt='[%(levelname)s][%(asctime)s]: %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    for handler in logger.handlers:
        handler.setFormatter(formatter)
    return logger

@hydra.main(config_path="./configs", config_name="config", version_base=None)
def run_training(cfg):
    # Setup logging
    logger = setup_logging()
    
    logger.info(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")
    print(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")

    # 1. Dataset and dataloaders
    image_files = glob.glob(os.path.join(cfg.paths.dataset_dir, "*.png"))
    original_targets = [x.split("/")[-1][:-4].replace("-copy", "") for x in image_files]
    targets = [[c for c in x] for x in original_targets]
    targets_flat = [c for clist in targets for c in clist]
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(targets_flat)
    targets_encoded = [label_encoder.transform(x) for x in targets]
    targets_encoded = np.array(targets_encoded)
    targets_encoded = targets_encoded + 1
    
    (train_imgs, 
    test_imgs, 
    train_targets, 
    test_targets, 
    _, 
    test_original_targets) = model_selection.train_test_split(image_files, targets_encoded, original_targets, test_size=0.1, random_state=42)

    train_dataset = dataset.ClassificationDataset(
        image_paths=train_imgs,
        targets=train_targets,
        resize=(cfg.processing.image_height, cfg.processing.image_width),
        grayscale=cfg.model.grayscale
    )

    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True,
    )

 
    test_dataset = dataset.ClassificationDataset(
        image_paths=test_imgs,
        targets=test_targets,
        resize=(cfg.processing.image_height, cfg.processing.image_width),
        grayscale=cfg.model.grayscale
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False,
    )

    print(f"Dataset number of classes: {len(label_encoder.classes_)}")
    print(f"Classes are: {label_encoder.classes_}")
    logging.info(f"Dataset number of classes: {len(label_encoder.classes_)}")
    logging.info(f"Classes are: {label_encoder.classes_}")
    
    # 2. Setup model, optim and scheduler
    device = cfg.processing.device
    model = CRNN(dims=cfg.model.dims, 
        num_chars=len(label_encoder.classes_), 
        use_attention=cfg.model.use_attention, 
        use_ctc=cfg.model.use_ctc,
        grayscale=cfg.model.grayscale)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=1, rho=0.85, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )

    best_acc = 0.0
    train_loss_data = []
    valid_loss_data = []
    accuracy_data = []
   
    # This is the same list as training, but with the '∅' token, that denotes blank for ctc, or pad for cross_entropy
    classes = ['∅']
    classes.extend(label_encoder.classes_)

    # 3. Training and logging
    if not os.path.exists('logs'): os.makedirs('logs')
    start = datetime.now()
    for epoch in range(cfg.training.num_epochs):
        # Train
        train_loss = engine.train_fn(model, train_loader, optimizer, device)
        train_loss_data.append(train_loss)
        # Eval
        valid_preds, test_loss = engine.eval_fn(model, test_loader, device)
        valid_loss_data.append(test_loss)
        # Eval + decoding for logging purposes
        valid_captcha_preds = []
        
        for vp in valid_preds:
            if model.use_ctc:
                current_preds = decode_predictions(vp, classes)
            else:
                current_preds = decode_padded_predictions(vp, classes)
            valid_captcha_preds.extend(current_preds)

        # Logging
        combined = list(zip(test_original_targets, valid_captcha_preds))
        if cfg.bools.VIEW_INFERENCE_WHILE_TRAINING:
            table = Table(show_header=True, header_style="hot_pink")
            table.add_column("Ground Truth", width=12)
            table.add_column("Predicted")
            table.border_style = "bright_yellow"
            table.columns[0].style = "violet"
            table.columns[1].style = "grey93"
            for idx in combined[:]:
                if cfg.bools.DISPLAY_ONLY_WRONG_PREDICTIONS:
                    if idx[0] != idx[1]:
                        table.add_row(idx[0]
                        ,idx[1])
                else: 
                    table.add_row(idx[0]
                    ,idx[1])
            console.print(table)
        
        #print(valid_captcha_preds)
        accuracy = metrics.accuracy_score(test_original_targets, valid_captcha_preds)
    
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
            logger.info(f"New best accuracy achieved at epoch {epoch}. Best accuracy now is: {best_acc}")
            best_model_wts = copy.deepcopy(model.state_dict())
            if cfg.bools.SAVE_CHECKPOINTS:
                torch.save(model, f"logs/checkpoint-{(best_acc*100):.2f}.pth")

        scheduler.step(test_loss)
        table.add_row(str(epoch),
        str(train_loss),
        str(test_loss),
        str(accuracy), 
        str(best_acc))
        console.print(table)
        logger.info(f"Epoch {epoch}:    Train loss: {train_loss}    Test loss: {test_loss}    Accuracy: {accuracy}")

    # 4. Save model + logging and plotting
    logger.info(f"Finished training. Best Accuracy was: {(best_acc*100):.2f}%")

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), cfg.paths.save_model_as)

    logger.info(f"Saving model on {cfg.paths.save_model_as}\nTraining time: {datetime.now()-start}")
    
    plot_losses(train_loss_data, valid_loss_data)
    plot_acc(accuracy_data)
   

if __name__ == "__main__":
    try:
        torch.cuda.empty_cache()
        run_training()
    except Exception:
        console.print_exception()

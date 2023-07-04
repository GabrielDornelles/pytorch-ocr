import os
import copy
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import torch
from sklearn import metrics
from rich.console import Console
from rich import print

import engine
from utils.logging_config import setup_logging, general_table, predictions_table
from models.crnn import CRNN
from utils.plot import plot_acc, plot_losses
from utils.model_decoders import decode_predictions, decode_padded_predictions
from utils.data_loading import build_dataloaders


# Setup rich console
console = Console()


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def run_training(cfg):
    # Setup logging
    logger = setup_logging()

    logger.info(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")
    print(f"Configurations:\n{OmegaConf.to_yaml(cfg)}")

    # 1. Dataset and dataloaders
    train_loader, test_loader, test_original_targets, classes = build_dataloaders(cfg)

    print(f"Dataset number of classes: {len(classes)}")
    print(f"Classes are: {classes}")
    logger.info(f"Dataset number of classes: {len(classes)}")
    logger.info(f"Classes are: {classes}")

    # 2. Setup model, optim and scheduler
    device = cfg.processing.device
    model = CRNN(
        resolution=(cfg.processing.image_width, cfg.processing.image_height),
        dims=cfg.model.dims,
        num_chars=len(classes),
        use_attention=cfg.model.use_attention,
        use_ctc=cfg.model.use_ctc,
        grayscale=cfg.model.gray_scale,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True)

    best_acc = 0.0
    train_loss_data = []
    valid_loss_data = []
    accuracy_data = []

    # This is the same list of characters from dataset, but with the '∅' token
    # which denotes blank for ctc, or pad for cross_entropy
    training_classes = ["∅"]
    training_classes.extend(classes)

    # 3. Training and logging
    if not os.path.exists("logs"):
        os.makedirs("logs")
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
                print(vp.shape)
                current_preds = decode_predictions(vp, training_classes)
            else:
                print(vp)
                current_preds = decode_padded_predictions(vp, training_classes)
            valid_captcha_preds.extend(current_preds)

        # Logging
        combined = list(zip(test_original_targets, valid_captcha_preds))
        if cfg.bools.VIEW_INFERENCE_WHILE_TRAINING:
            table = predictions_table()
            for idx in combined:
                if cfg.bools.DISPLAY_ONLY_WRONG_PREDICTIONS:
                    if idx[0] != idx[1]:
                        table.add_row(idx[0], idx[1])
                else:
                    table.add_row(idx[0], idx[1])
            console.print(table)

        accuracy = metrics.accuracy_score(test_original_targets, valid_captcha_preds)
        accuracy_data.append(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            logger.info(f"New best accuracy achieved at epoch {epoch}. Best accuracy now is: {best_acc}")
            best_model_wts = copy.deepcopy(model.state_dict())
            if cfg.bools.SAVE_CHECKPOINTS:
                torch.save(model, f"logs/checkpoint-{(best_acc*100):.2f}.pth")

        scheduler.step(test_loss)
        table = general_table()
        table.add_row(str(epoch), str(train_loss), str(test_loss), str(accuracy), str(best_acc))
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

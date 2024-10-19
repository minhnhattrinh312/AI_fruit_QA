from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
from torch.utils.data import DataLoader
import sys
import pandas as pd


from classification import *
from lightning.pytorch.loggers import WandbLogger
import os
import wandb
import torch._dynamo
torch._dynamo.config.suppress_errors = True

torch.set_float32_matmul_precision("high")


# Main function
if __name__ == "__main__":
    # Loop over the folds
    for fold in cfg.TRAIN.FOLDS:
        print("train on fold", fold)

        save_dir = f"{cfg.DIRS.SAVE_DIR}/{cfg.TRAIN.TASK}_fold_{fold}/"
        os.makedirs(save_dir, exist_ok=True)
        x_train = pd.read_csv(f"k_fold_data/x_train_fold{fold}.csv")
        y_train = pd.read_csv(f"k_fold_data/y_{cfg.TRAIN.TASK}_train_fold{fold}.csv")
        x_test = pd.read_csv(f"k_fold_data/x_test_fold{fold}.csv")
        y_test = pd.read_csv(f"k_fold_data/y_{cfg.TRAIN.TASK}_test_fold{fold}.csv")
        # get dataframe train and test

        train_loader = SpectralDataset(x_train, y_train)
        test_loader = SpectralDataset(x_test, y_test)
        # Define data loaders for the training and test data

        train_dataset = DataLoader(
            train_loader,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            pin_memory=True,
            shuffle=True,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            drop_last=True,
            prefetch_factor=cfg.TRAIN.PREFETCH_FACTOR,
        )
        test_dataset = DataLoader(
            test_loader,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            prefetch_factor=cfg.TRAIN.PREFETCH_FACTOR,
        )

        model = BasicModel()
        classifier = Classifier(model, cfg.OPT.LEARNING_RATE, cfg.OPT.FACTOR_LR, cfg.OPT.PATIENCE_LR)

        # Initialize a ModelCheckpoint callback to save the model weights after each epoch
        check_point_mse = ModelCheckpoint(
            save_dir,
            filename="{val_loss:0.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=cfg.TRAIN.SAVE_TOP_K,
            verbose=True,
            save_weights_only=True,
            auto_insert_metric_name=False,
            save_last=False,
        )

        # Initialize a LearningRateMonitor callback to log the learning rate during training
        lr_monitor = LearningRateMonitor(logging_interval="step")
        # Initialize a EarlyStopping callback to stop training if the validation loss does not improve for a certain number of epochs
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg.OPT.PATIENCE_ES,
            verbose=True,
            strict=False,
        )
        # If wandb_logger is True, create a WandbLogger object
        if cfg.TRAIN.WANDB:
            wandb_logger = WandbLogger(
                project="AI_fruit_QA",
                name=f"{cfg.TRAIN.TASK}_fold_{fold}",
                group=f"{cfg.TRAIN.TASK}",
                resume="allow",
            )
            callbacks = [check_point_mse, early_stopping, lr_monitor]
        else:
            wandb_logger = False
            callbacks = [check_point_mse, early_stopping]

        # Define a dictionary with the parameters for the Trainer object
        PARAMS_TRAINER = {
            "accelerator": cfg.SYS.ACCELERATOR,
            "devices": cfg.SYS.DEVICES,
            "benchmark": True,
            "enable_progress_bar": True,
            # "overfit_batches" :5,
            "logger": wandb_logger,
            "callbacks": callbacks,
            "log_every_n_steps": 1,
            "num_sanity_val_steps": 0,
            "max_epochs": cfg.TRAIN.EPOCHS,
            "precision": cfg.SYS.MIX_PRECISION,
        }

        # Initialize a Trainer object with the specified parameters
        trainer = Trainer(**PARAMS_TRAINER)
        # Get a list of file paths for all non-hidden files in the SAVE_DIR directory
        checkpoint_paths = [save_dir + f for f in os.listdir(save_dir) if not f.startswith(".")]
        checkpoint_paths.sort()
        # If there are checkpoint paths and the load_checkpoint flag is set to True
        if checkpoint_paths and cfg.TRAIN.LOAD_CHECKPOINT:
            # Select the second checkpoint in the list (index 0)
            checkpoint = checkpoint_paths[cfg.TRAIN.IDX_CHECKPOINT]
            print(f"load checkpoint: {checkpoint}")
            # Load the model weights from the selected checkpoint
            classifier = Classifier.load_from_checkpoint(
                checkpoint_path=checkpoint,
                model=model,
                class_weight=cfg.DATA.CLASS_WEIGHT,
                num_classes=cfg.DATA.NUM_CLASS,
                learning_rate=cfg.OPT.LEARNING_RATE,
                factor_lr=cfg.OPT.FACTOR_LR,
                patience_lr=cfg.OPT.PATIENCE_LR,
            )

        # Train the model using the train_dataset and test_dataset data loaders
        trainer.fit(classifier, train_dataset, test_dataset)

        wandb.finish()

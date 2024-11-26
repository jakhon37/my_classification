import os
import torch
import torch.optim as optim
from data import get_dataloaders
# from models import *
import models
# from losses import * 
import losses
from config import config
from utils import save_checkpoint, calculate_accuracy, setup_logging, plot_training_curves
import logging
import time



def train(config):
    # Setup unique output directory based on timestamp
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join(config.logging.save_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    setup_logging(os.path.join(output_dir, 'training.log'))
    logger = logging.getLogger('Training')

    # Data loaders
    train_loader, val_loader = get_dataloaders(config)

    # Model
    model_ = getattr(models, config.model.name)
    model = model_(config.model.params.num_classes).to(config.device)
    logger.info(f"Using model: {config.model.name}")

    # Loss function
    # loss_ = globals()[config.loss.name]
    loss_ = getattr(losses, config.loss.name)
    criterion = loss_()
    logger.info(f"Using loss function: {config.loss.name}")

    # Optimizer
    # optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    optimizer_ = getattr(torch.optim, config.training.optimizer)
    optimizer = optimizer_(model.parameters(), lr=config.training.learning_rate, **config.training.optimizer_params)
    logger.info(f"Using optimizer: Adam with learning rate {config.training.learning_rate}")


    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(config.training.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0

        for images, labels in train_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            running_accuracy += calculate_accuracy(outputs, labels).item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = (running_accuracy / len(train_loader)) * 100
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        logger.info(f"Epoch [{epoch+1}/{config.training.epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_accuracy = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.device), labels.to(config.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Statistics
                val_running_loss += loss.item()
                val_running_accuracy += calculate_accuracy(outputs, labels).item()

        val_epoch_loss = val_running_loss / len(val_loader)
        val_epoch_accuracy = (val_running_accuracy / len(val_loader)) * 100
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        logger.info(f"Epoch [{epoch+1}/{config.training.epochs}], Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.2f}%")

        # Save the best model
        if val_epoch_accuracy > best_val_accuracy:
            best_val_accuracy = val_epoch_accuracy
            if config.logging.save_all_best_ckpt:
                best_model_path = os.path.join(output_dir, f"best_model_epoch_{epoch+1}.pth")
                save_checkpoint({'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                                best_model_path)
                logger.info(f"Best model saved with accuracy: {best_val_accuracy:.2f}% at {best_model_path}")

        # Save checkpoint every epoch (optional)
        if config.logging.save_all_ckpt:
            epoch_model_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pth")
            save_checkpoint({'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                            epoch_model_path)
            logger.info(f"Model checkpoint saved at {epoch_model_path}")

    # Plot and save training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, output_dir)
    logger.info(f"Training curves saved in {output_dir}")

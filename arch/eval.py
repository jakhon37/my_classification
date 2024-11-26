import os
import torch
from data import get_dataloaders
from models import *
from losses import *
from config import config
from utils import calculate_accuracy, load_checkpoint, setup_logging
import logging
import glob

def validate():
    # Setup logging
    setup_logging('validation.log')
    logger = logging.getLogger('Validation')

    # Data loaders
    _, val_loader = get_dataloaders(config['batch_size'], config['data_dir'])

    # Model
    model_class = globals()[config['model_name']]
    model = model_class(config['num_classes']).to(config['device'])
    logger.info(f"Using model: {config['model_name']}")

    # Load the latest best model
    list_of_files = glob.glob(os.path.join(config['save_dir'], '*', 'best_model_epoch_*.pth'))
    if not list_of_files:
        logger.error("No best model found in checkpoints.")
        return
    latest_model_path = max(list_of_files, key=os.path.getctime)
    model, _ = load_checkpoint(latest_model_path, model)
    logger.info(f"Loaded model from {latest_model_path}")

    # Loss function
    loss_class = globals()[config['loss_name']]
    criterion = loss_class()
    logger.info(f"Using loss function: {config['loss_name']}")

    # Evaluation
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config['device']), labels.to(config['device'])
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, labels).item()

    avg_loss = val_loss / len(val_loader)
    avg_accuracy = (val_accuracy / len(val_loader)) * 100
    logger.info(f"Validation Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")

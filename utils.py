import os
import torch
import logging
from omegaconf import OmegaConf
import time



def load_config(config_file='config.yaml'):
    config_file = os.path.abspath(config_file)
    try:
        config = OmegaConf.load(config_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_file} not found.")
    config.experiment.timestamp = time.strftime('%Y%m%d-%H%M%S')
    return config

def setup_logging(log_file):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_file,
                        filemode='w')
    # Also print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer=None):
    checkpoint = torch.load(filename, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def calculate_accuracy(outputs, targets):
    _, preds = outputs.max(1)
    correct = preds.eq(targets).sum()
    accuracy = correct.float() / targets.size(0)
    return accuracy


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, output_dir):
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure()
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()

    # Plot Accuracy
    plt.figure()
    plt.plot(epochs, train_accuracies, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    acc_plot_path = os.path.join(output_dir, 'accuracy_curve.png')
    plt.savefig(acc_plot_path)
    plt.close()
    
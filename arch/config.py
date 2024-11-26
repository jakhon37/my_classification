
import torch 

config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_classes': 2,
    'data_dir': '/home/aivar/deep/classify/datasets/dog&cat',
    'save_dir': './checkpoints',
    'model_name': 'ModelC',  # Change to switch models
    'loss_name': 'LossA',    # Change to switch loss functions
}

if __name__=="__main__":
    import os 
    directory = os.path.join(config['data_dir'], 'train')
    print(f'directory: {directory}')
    # directory = '/home/aivar/deep/classify/datasets/dog&cat/train'
    print(f'directory: {directory}')
    
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    print(f'classes: {classes}')
# home/aivar/deep/classify/datasets/dog&cat/train
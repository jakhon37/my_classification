import argparse
from train import train
# from eval import validate
from utils import load_config
from omegaconf import OmegaConf
import random
import numpy as np
import torch



def validate_config(config):
    assert config.training.batch_size > 0, "Batch size must be positive"
    assert config.training.learning_rate > 0, "Learning rate must be positive"
    # Add other checks as needed

def set_seed(config):
    seed = config.experiment.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def main():
    parser = argparse.ArgumentParser(description='Classification Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validate'], help='Mode to run the script')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')

    # args = parser.parse_args()
    args, unknown_args = parser.parse_known_args()

    # Load and merge configurations
    config = load_config(args.config)
    cli_conf = OmegaConf.from_cli(unknown_args)
    config = OmegaConf.merge(config, cli_conf)

    # Validate configurations
    validate_config(config)
    

    set_seed(config)

    if args.mode == 'train':
        train(config)
    # elif args.mode == 'validate':
    #     validate()
    else:
        print("Invalid mode selected. Choose 'train' or 'validate'.")

if __name__ == '__main__':
    main()

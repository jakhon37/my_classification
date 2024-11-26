import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from omegaconf import DictConfig, ListConfig


def get_transforms(config):
    # Function to retrieve and apply transformations dynamically with optional params
    def parse_transform(aug_name, params=None):
        if not isinstance(aug_name, str):  # Ensure aug_name is a string
            aug_name = str(aug_name)
        transform_func = getattr(transforms, aug_name, None)
        if not transform_func:
            print(f"Warning: {aug_name} is not a valid transformation in torchvision.transforms")
            return None
        # Apply transformation with parameters if available
        return transform_func(**params) if params else transform_func()

    # Build transformations for both training and evaluation
    def build_transform_list(augmentation_list):
        transform_list = []
        for aug in augmentation_list:
            # print(f'aug: {aug}')
            # print(f'aug type: {type(aug)}')
            if isinstance(aug, DictConfig):  # Allows for param-based augmentation
                # print(f'aug is dict: {type(aug)}')
                # aug_name = list(aug.keys())[0]
                # aug_params = aug[aug_name]
                # transform = parse_transform(aug_name, aug_params)
                for aug_name, aug_params in aug.items():
                    transform = parse_transform(aug_name, aug_params)
                    if transform:
                        transform_list.append(transform)
            else:
                # print(f'aug is not dict: {type(aug)}')
                
                transform = parse_transform(aug)
            if transform:
                transform_list.append(transform)
        return transform_list

    # Training transformations
    train_augmentations = config.data.augmentations.train
    train_transform_list = build_transform_list(train_augmentations)
    train_transform = transforms.Compose(train_transform_list)

    # Evaluation transformations
    eval_augmentations = config.data.augmentations.eval
    eval_transform_list = build_transform_list(eval_augmentations)
    eval_transform = transforms.Compose(eval_transform_list)

    return train_transform, eval_transform


def get_datasets(config):
    train_transform, val_transform = get_transforms(config)
    train_dataset = datasets.ImageFolder(os.path.join(config.data.data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(config.data.data_dir, 'valid'), transform=val_transform)
    return train_dataset, val_dataset

def get_dataloaders(config):
    train_dataset, val_dataset = get_datasets(config)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=config.data.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader

# Training Code for Classification Problem

This repository contains a modular and extensible training codebase for classification problems using PyTorch. The code is organized to facilitate easy addition of new datasets, models, loss functions, and configurations. It supports training, validation, and inference, with features like logging, checkpointing, and plotting training curves.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Training Script](#running-the-training-script)
  - [Configuration](#configuration)
- [Adding a New Dataset](#adding-a-new-dataset)
- [Adding a New Model](#adding-a-new-model)
- [Adding a New Loss Function](#adding-a-new-loss-function)
- [Modifying Configurations](#modifying-configurations)
- [Logging and Checkpoints](#logging-and-checkpoints)
- [Visualizing Training Metrics](#visualizing-training-metrics)
- [Tips and Best Practices](#tips-and-best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

This training code is designed to be a flexible and extensible framework for classification tasks. It allows you to:

- Train models on your own datasets.
- Easily switch between different models and loss functions.
- Customize training configurations via a YAML file or command-line arguments.
- Automatically log training progress and save the best model checkpoints.
- Visualize training and validation metrics.

---

## Project Structure

The project is organized as follows:

```
project/
├── data.py
├── models/
│   ├── __init__.py
│   ├── model_a.py
│   ├── model_b.py
│   ├── model_c.py
│   └── ...           # Additional models
├── losses/
│   ├── __init__.py
│   ├── loss_a.py
│   ├── loss_b.py
│   └── ...           # Additional loss functions
├── train.py
├── val.py
├── inference.py
├── config.py
├── utils.py
├── main.py
├── requirements.txt
├── config.yaml
└── checkpoints/      # Directory for saved models and logs
```

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/project.git
   cd project
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **Contents of `requirements.txt`:**

   ```
   torch
   torchvision
   numpy
   matplotlib
   PyYAML
   OmegaConf
   ```

---

## Usage

### Running the Training Script

To start training, run:

```bash
python main.py
```

By default, this will use the configurations specified in `config.yaml`. You can override configurations using command-line arguments.

### Configuration

The training script uses a YAML configuration file (`config.yaml`) to manage settings. You can modify this file or override specific parameters via command-line arguments.

**Example Command-Line Overrides:**

```bash
python main.py training.batch_size=64 training.learning_rate=0.0005 experiment.name='new_experiment'
```

**Sample `config.yaml`:**

```yaml
experiment:
  name: 'experiment_1'
  description: 'Baseline model with default settings'
  seed: 42
  timestamp: null

model:
  name: 'ModelC'
  params:
    num_classes: 2

loss:
  name: 'LossA'

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 20
  optimizer: 'Adam'
  optimizer_params:
    weight_decay: 0.0001
  scheduler:
    name: 'StepLR'
    params:
      step_size: 10
      gamma: 0.1

data:
  data_dir: './datasets/dog&cat'
  size: &size 224
  num_workers: 4
  augmentations:
    train:
      - Resize:
          size: [*size, *size]
      - RandomHorizontalFlip
      - RandomCrop:
          size: *size
      - ToTensor
      - Normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
    eval:
      - Resize:
          size: [*size, *size]
      - CenterCrop:
          size: *size
      - ToTensor
      - Normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]

logging:
  save_dir: './checkpoints'
  log_interval: 10
  tensorboard: False

device: 'cuda'  # or 'cpu'
```

---

## Adding a New Dataset

To use your own dataset, follow these steps:

1. **Organize Your Dataset**

   Structure your dataset in the following format, suitable for `torchvision.datasets.ImageFolder`:

   ```
   data_dir/
   ├── train/
   │   ├── class1/
   │   │   ├── img1.jpg
   │   │   ├── img2.jpg
   │   └── class2/
   │       ├── img3.jpg
   │       ├── img4.jpg
   └── valid/
       ├── class1/
       │   ├── img5.jpg
       └── class2/
           ├── img6.jpg
   ```

2. **Update `config.yaml`**

   Set the `data_dir` in the configuration file to point to your dataset:

   ```yaml
   data:
     data_dir: '/path/to/your/data_dir'
     # ... other data configurations
   ```

3. **Adjust Number of Classes**

   Update `num_classes` in `config.yaml` to match the number of classes in your dataset:

   ```yaml
   model:
     name: 'YourModel'
     params:
       num_classes: 3  # Set this to your number of classes
   ```

---

## Adding a New Model

To add a new model architecture:

1. **Create a New Model File**

   Create a Python file in the `models/` directory, e.g., `models/your_model.py`.

2. **Implement Your Model**

   Define your model class inheriting from `torch.nn.Module`:

   ```python
   # models/your_model.py
   import torch.nn as nn

   class YourModel(nn.Module):
       def __init__(self, num_classes):
           super(YourModel, self).__init__()
           # Define your layers
           self.features = nn.Sequential(
               # Add your layers here
           )
           self.classifier = nn.Linear(in_features, num_classes)

       def forward(self, x):
           x = self.features(x)
           x = x.view(x.size(0), -1)
           x = self.classifier(x)
           return x
   ```

3. **Register Your Model**

   Add your model to `models/__init__.py`:

   ```python
   from .your_model import YourModel
   # Existing imports
   from .model_a import ModelA
   from .model_b import ModelB
   from .model_c import ModelC
   ```

4. **Update Configuration**

   In `config.yaml`, set the model name and parameters:

   ```yaml
   model:
     name: 'YourModel'
     params:
       num_classes: 3
       # Add other model-specific parameters if any
   ```

---

## Adding a New Loss Function

To implement a custom loss function:

1. **Create a New Loss File**

   Create a Python file in the `losses/` directory, e.g., `losses/your_loss.py`.

2. **Implement Your Loss Function**

   Define your loss class inheriting from `torch.nn.Module`:

   ```python
   # losses/your_loss.py
   import torch.nn as nn

   class YourLoss(nn.Module):
       def __init__(self):
           super(YourLoss, self).__init__()
           # Initialize your loss function components

       def forward(self, outputs, targets):
           # Compute and return the loss
           loss = # Your loss computation
           return loss
   ```

3. **Register Your Loss Function**

   Add your loss to `losses/__init__.py`:

   ```python
   from .your_loss import YourLoss
   # Existing imports
   from .loss_a import LossA
   from .loss_b import LossB
   ```

4. **Update Configuration**

   In `config.yaml`, set the loss name:

   ```yaml
   loss:
     name: 'YourLoss'
   ```

---

## Modifying Configurations

Configurations are managed via the `config.yaml` file and can be overridden using command-line arguments.

### Common Modifications

- **Batch Size and Learning Rate**

  ```yaml
  training:
    batch_size: 64
    learning_rate: 0.0005
  ```

- **Optimizer and Scheduler**

  ```yaml
  training:
    optimizer: 'SGD'
    optimizer_params:
      momentum: 0.9
    scheduler:
      name: 'CosineAnnealingLR'
      params:
        T_max: 50
  ```

- **Data Augmentations**

  Adjust the augmentations in the `data` section:

  ```yaml
  data:
    augmentations:
      train:
        - Resize:
            size: [256, 256]
        - RandomResizedCrop:
            size: 224
        - RandomHorizontalFlip
        - ColorJitter:
            brightness: 0.2
            contrast: 0.2
            saturation: 0.2
            hue: 0.1
        - ToTensor
        - Normalize:
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
  ```

### Using Command-Line Overrides

You can override any configuration parameter via command-line:

```bash
python main.py training.batch_size=128 model.name='AnotherModel' data.num_workers=8
```

---

## Logging and Checkpoints

- **Logging**

  - Logs are saved in the `checkpoints/` directory, under a subdirectory named with a timestamp.
  - Training progress, including losses and accuracies, is logged in a `training.log` file.

- **Checkpoints**

  - Model checkpoints are saved after each epoch in the corresponding subdirectory.
  - The best model (based on validation accuracy) is saved separately as `best_model_epoch_X.pth`.

- **TensorBoard (Optional)**

  - If `logging.tensorboard` is set to `True` in `config.yaml`, TensorBoard logs will be saved.
  - Launch TensorBoard using:

    ```bash
    tensorboard --logdir=checkpoints/
    ```

---

## Visualizing Training Metrics

After training, loss and accuracy curves are saved in the output directory.

- **Loss Curve**: `loss_curve.png`
- **Accuracy Curve**: `accuracy_curve.png`

These plots show the training and validation metrics over epochs, helping you assess the model's performance.

---

## Tips and Best Practices

- **Set Random Seeds for Reproducibility**

  ```yaml
  experiment:
    seed: 42
  ```

- **Use Appropriate Data Augmentations**

  Tailor the augmentations to your dataset to improve generalization.

- **Monitor for Overfitting**

  Keep an eye on the training and validation curves. If validation performance degrades while training performance improves, consider regularization techniques.

- **Adjust Learning Rate**

  Experiment with different learning rates and schedulers to find the optimal settings.

- **Utilize GPU Acceleration**

  Ensure `device` is set to `'cuda'` if a GPU is available.

---

## Troubleshooting

- **Error: `_tkinter.TclError: couldn't connect to display`**

  - **Solution**: Set `matplotlib` to use a non-interactive backend. This is already handled in `utils.py` by setting:

    ```python
    import matplotlib
    matplotlib.use('Agg')
    ```

- **TypeError when Parsing Augmentations**

  - **Solution**: Ensure that `data.py` correctly handles `DictConfig` types from `OmegaConf`. This is addressed by importing `DictConfig` and adjusting type checks.

- **RuntimeError Due to Tensor Shape Mismatch**

  - **Solution**: Verify that the output of your model's feature extractor matches the input size of the classifier. Use adaptive pooling or dynamically compute the input features.

- **Invalid Transformation Warnings**

  - **Solution**: Ensure that transformation names and parameters in `config.yaml` match those expected by `torchvision.transforms`.

---

## Contributing

Contributions are welcome! If you'd like to add new features or fix bugs, please:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.

---

By following this guide, you should be able to effectively use and extend this training codebase for your classification tasks. If you have any questions or run into issues, feel free to open an issue on the repository or reach out for assistance.
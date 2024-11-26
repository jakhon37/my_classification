import torch.nn as nn

class ModelA(nn.Module):
    def __init__(self, num_classes):
        super(ModelA, self).__init__()
        # Define layers
        self.layer = nn.Linear(512, num_classes)

    def forward(self, x):
        # Define forward pass
        return self.layer(x)

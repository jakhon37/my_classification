import torch.nn as nn

class ModelB(nn.Module):
    def __init__(self, num_classes):
        super(ModelB, self).__init__()
        # Define a different architecture
        self.layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layer(x)

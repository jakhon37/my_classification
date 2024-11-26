import torch.nn as nn

class ModelC(nn.Module):
    def __init__(self, num_classes):
        super(ModelC, self).__init__()
        # Example: Simple CNN
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Add more layers as needed
        )
        self.classifier = nn.Linear(64 * 112 * 112, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

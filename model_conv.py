import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, num_input_channels):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_input_channels, 16, (3, 3), stride=(1, 1), padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), (2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), (2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, (1, 1), (1, 1)),
            nn.ReLU()
        )
        self.model2 = nn.Sequential(
            nn.Linear(23 * 23 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        return self.model2(out)

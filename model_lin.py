import torch.nn as nn
import torch


class LinModel(nn.Module):
    def __init__(self):
        super(LinModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100*100, 2*1000),
            nn.ReLU(),
            nn.Linear(2000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        return self.model(x)

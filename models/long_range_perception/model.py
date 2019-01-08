import torch
import torch.nn as nn
import torch.optim as optim

def conv_act_max(act='relu', *args, **kwargs):
    activation = nn.ModuleDict({
                        'relu': nn.ReLU(),
                        'lrelu': nn.LeakyReLU()
                })[act]

    return nn.Sequential(
        nn.Conv2d(*args, **kwargs, kernel_size=3, padding=1),
        activation,
        nn.MaxPool2d(kernel_size=2)
    )

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.encoder = nn.Sequential(
            *conv_act_max(in_channels=in_channels, out_channels=10),
            *conv_act_max(in_channels=10, out_channels=10),
            *conv_act_max(in_channels=10, out_channels=8),
        )

        self.decoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=8*8*10, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=65*5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.decoder(x)

        return x

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        num_layers,
        activation="LeakyReLU",
        dropout=0.1,
        leaky_slope=0.01,
    ):
        super().__init__()

        act = {
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(leaky_slope),
            "GELU": nn.GELU(),
        }[activation]
        assert num_layers >= 1

        layers = [nn.Linear(in_channels, hidden_channels)]
        for _ in range(num_layers - 1):
            layers.append(act)
            if dropout:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(hidden_channels, hidden_channels))
        layers.append(act)
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

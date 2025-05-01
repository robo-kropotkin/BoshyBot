import torch
from torch import nn, optim


class QNet(nn.Module):
    def __init__(self, lr=0.01, input_dims=6, n_actions=5):
        super(QNet, self).__init__()
        self.input_dims = input_dims

        # Simplified network with fewer layers and better initialization
        self.model = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

        # Apply weight initialization
        self.model.apply(self.init_weights)

        # Use a gradient clipping optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        self.loss = nn.HuberLoss()  # Huber loss is more robust for DQN than MSE

        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        return self.model.forward(x)

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, nn.Linear):
            # Use a smaller initialization scale
            nn.init.xavier_uniform_(layer.weight, gain=0.01)
            # Initialize biases to small values
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.01)
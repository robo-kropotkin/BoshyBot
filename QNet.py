import torch
from torch import nn, optim

class QNet(nn.Module):
    def __init__(self, lr=0.03, input_dims=9, n_actions=16):
        super(QNet, self).__init__()
        self.input_dims = input_dims
        self.final_layer = nn.Linear(128, n_actions)
        nn.init.constant_(self.final_layer.bias, 0)
        self.model = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.8),
            self.final_layer
        )
        self.model.apply(self.init_weights)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0")
        self.to(self.device)

    def forward(self, x):
        return self.model.forward(x)

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, nn.Linear):
            print("Initializing weights.")
            nn.init.xavier_uniform_(layer.weight)

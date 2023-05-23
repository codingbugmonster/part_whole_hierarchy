import torch.nn as nn

class dae(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(dae, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

        self.fc1.weight.data.normal_(0, 1)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.normal_(0, 1)
        self.fc2.bias.data.fill_(0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


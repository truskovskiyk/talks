import torch.nn as nn
import torch.nn.functional as F


class NetFC(nn.Module):
    def __init__(self):
        super(NetFC, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return F.log_softmax(x, dim=1)

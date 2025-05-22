import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATClassifier(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 18085,
                 num_classes: int = 3,
                 hidden_channels: int = 16):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)  # Multi-head attention
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Graph-level representation
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

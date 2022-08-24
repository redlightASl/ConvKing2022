import torch
from torch import nn


class LinearStruct(nn.Module):
    def __init__(self, in_plane=100, out_plane=10):
        super(LinearStruct, self).__init__()
        self.fc = nn.Linear(in_features=in_plane,
                            out_features=out_plane,
                            bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == '__main__':
    in_data = torch.tensor([[[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 2, 1, 1, 2, 0, 0, 0, 0],
                              [0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0],
                              [0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]],
                           dtype=torch.float32)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LinearStruct(in_plane=12 * 12, out_plane=10)
    net.fc.weight.data = torch.full([12,12],1,dtype=torch.float)
    print(net.fc.weight.data.size())
    result = net(in_data)
    print(result.int())
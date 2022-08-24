import torch
import torch.nn as nn


class MaxPoolStruct(nn.Module):
    def __init__(self):
        super(MaxPoolStruct, self).__init__()
        self.mp1 = nn.MaxPool2d(kernel_size=2,stride=1,padding=0)

    def forward(self, x):
        x = self.mp1(x)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MaxPoolStruct().to(device)
    result = net(in_data)
    print(result.int())
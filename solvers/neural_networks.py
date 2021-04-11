import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(MultiLayerPerceptron, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.dense1 = nn.Linear(input_size, 64)
        self.activation1 = nn.ReLU()
        self.dense2 = nn.Linear(64, 32)
        self.activation2 = nn.ReLU()
        self.dense3 = nn.Linear(32, 16)
        self.activation3 = nn.ReLU()
        self.output = nn.Linear(16, output_size)
        self.softmax = nn.Softmax(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ret = self.dense1(x)
        ret = self.activation1(ret)
        ret = self.dense2(ret)
        ret = self.activation2(ret)
        ret = self.dense3(ret)
        ret = self.activation3(ret)
        ret = self.output(ret)
        if self.output_size > 1:
            ret = self.softmax(ret)
        return ret

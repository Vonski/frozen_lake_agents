import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """
    Simple multilayer perceptron implemented in torch.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Creates multilayer perceptron of given input and output sizes.

        Args:
            input_size: Number of inputs.
            output_size: Number of neurons in final layer.
        """
        super(MultiLayerPerceptron, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.dense1 = nn.Linear(input_size, 64)
        self.activation1 = nn.ReLU()
        # self.dense2 = nn.Linear(64, 64)
        # self.activation2 = nn.ReLU()
        # self.dense3 = nn.Linear(64, 64)
        # self.activation3 = nn.ReLU()
        self.output = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation function.

        Args:
            x: Input batch.

        Returns:
            Output of MLP for a given batch.
        """
        ret = self.dense1(x)
        ret = self.activation1(ret)
        # ret = self.dense2(ret)
        # ret = self.activation2(ret)
        # ret = self.dense3(ret)
        # ret = self.activation3(ret)
        ret = self.output(ret)
        # if self.output_size > 1:
        #     ret = self.softmax(ret)
        return ret

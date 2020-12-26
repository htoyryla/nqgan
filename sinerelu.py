import torch
import torch.nn as nn


def sineReLU_(input, eps = 0.01):

    return (input > 0).float() * input + (input <= 0).float() * eps * (torch.sin(input) - torch.cos(input))

class SineReLU(nn.Module):

    def __init__(self, epsilon = 0.01):

        super().__init__()
        self.epsilon = epsilon

    def forward(self, input):

        return sineReLU_(input, self.epsilon)

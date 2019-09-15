import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable


class CBN(nn.Module):
    """
  A conditional batch normalization layer from
  'Learning Visual Reasoning Without Strong Priors'
  """

    def __init__(self, epsilon=1e-5):
        super(CBN, self).__init__()
        self.epsilon = epsilon

    def forward(self, x: Variable, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)

        n, c, h, w = x.size()
        x_flat = x.view(n, c * h * w)

        mu = x_flat.mean(1)
        var = x_flat.std(1)

        print(x_flat.size(), mu.size(), var.size())
        x_norm = (x_flat - mu) / np.sqrt(var + self.epsilon)

        return (gammas * x_norm) + betas

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

        print(x.size(), gammas.size(), betas.size())

        n, c, h, w = x.size()
        x_flat = x.view(n, c * h * w)

        mu = x_flat.mean(1).expand_as(x_flat)
        var = x_flat.std(1).expand_as(x_flat)

        x_norm = (x_flat - mu) / (var + self.epsilon).sqrt()

        return (gammas * x_norm.view(n,c,h,w)) + betas

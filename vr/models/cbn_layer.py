import numpy as np
import torch.nn as nn
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

        print(x.size())
        print(gammas.size())
        print(betas.size())
        n, c, h, w = x.size()
        x_flat = x.reshape(n, c * h * w)

        mu = np.mean(x_flat, axis=0)
        var = np.var(x_flat, axis=0)
        x_norm = (x_flat - mu) / np.sqrt(var + self.epsilon)

        return (gammas * x_norm) + betas

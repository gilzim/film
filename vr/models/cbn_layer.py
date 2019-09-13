import numpy as np
import torch.nn as nn

class CBN(nn.Module):
    """
  cbn
  """

    def __init__(self, epsilon=1e-5):
        super(CBN, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)

        n, c, h, w = x.shape
        x_flat = x.reshape(n, c * h * w)

        mu = np.mean(x_flat, axis=0)
        var = np.var(x_flat, axis=0)
        x_norm = (x_flat - mu) / np.sqrt(var + self.epsilon)

        out = gammas * x_norm + betas

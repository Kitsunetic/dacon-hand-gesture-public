import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    https://dacon.io/competitions/official/235585/codeshare/1796
    """

    def __init__(self, gamma=2.0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # print(self.gamma)
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

from typing import Tuple, Union

import math
import torch
from torch import nn


class Coefficients(nn.Module):
    def __init__(
        self, 
        num_coefs: int, 
        target_value: float, 
        coef_range: Tuple[float, float] = (0.0, 1.0),
        init_coef: float = 1.0
    ):
        super().__init__()
        self.num_coefs = num_coefs
        self.target_value = target_value
        self.log_coef_range = (math.log(coef_range[0] + 1e-8), math.log(coef_range[1]))
        self.log_init_coef = math.log(max(init_coef, 1e-3))
        self.log_coefs = nn.Parameter(torch.as_tensor(data=self.log_init_coef, dtype=torch.float32).repeat(num_coefs))
    
    def __len__(self) -> int:
        return self.num_coefs
    
    def forward(self, coef_idxs: Union[torch.Tensor, None] = None, clip_coef: bool = True):
        """ Compute the coefficients for a batch of inputs. """
        log_coefs = self.log_coefs
        if (coef_idxs is not None) and (self.num_coefs > 1):
            log_coefs = log_coefs[coef_idxs]
        if clip_coef:
            log_coefs = torch.clamp(log_coefs, *self.log_coef_range)
        log_coefs = log_coefs.view(-1, 1)
        return log_coefs

    def reset(self):
        self.log_coefs.data.copy_(torch.full_like(self.log_coefs.data, fill_value=self.log_init_coef))
        return self
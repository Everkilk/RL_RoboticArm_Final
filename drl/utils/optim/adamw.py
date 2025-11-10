from typing import Tuple, Optional, Union, Any

import torch
from torch.optim import AdamW
from torch.optim.optimizer import _get_scalar_dtype

from drl.utils.optim.base import Optimizer, Params
from collections import defaultdict

class AdamWOptimizer(AdamW, Optimizer):
    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        polyak: float = 5e-3,
        max_norm: Optional[float] = None,
        accumulation: Optional[int] = 1,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= polyak:
            raise ValueError(f"Invalid polyak rate: {polyak}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            polyak=polyak,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
            decoupled_weight_decay=True,  # Required for PyTorch 2.0+ AdamW
        )
        Optimizer.__init__(self=self, params=params, defaults=defaults, max_norm=max_norm, accumulation=accumulation)
        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", True)  # Required for PyTorch 2.0+
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(is_fused=fused),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def step(self, closure: Union[Any, None]=None):
        """ Performs a single optimization step. """
        return super().step(closure=closure)

    def reset(self):
        """ Reset the state of the optimizer."""
        if len(self.state) == 0:
            return
        self.state = defaultdict(dict)
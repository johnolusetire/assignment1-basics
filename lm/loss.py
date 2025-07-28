from typing import Any, Callable, Dict
import torch
from typing import Iterable, Tuple, Optional
from torch import Tensor
import math

def log_softmax(x: torch.Tensor, dim: int):
    x = x - torch.max(x, dim=dim, keepdim=True).values
    return x - torch.log(torch.exp(x).sum(dim=dim, keepdim=True)) + 1e-8

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    log_probs = log_softmax(logits, dim=-1)
    correct_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1))
    return -correct_probs.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[Tensor] | Iterable[Dict[str, Any]] | Iterable[Tuple[str, Tensor]], lr: float, 
                 weight_decay: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}")
        defaults = {"lr": lr,
                    "betas": betas,
                    "weight_decay": weight_decay,
                    "epsilon": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            weight_decay = group["weight_decay"]
            epsilon = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]  

                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p, device = p.device)
                    state["v"] = torch.zeros_like(p, device = p.device)
                
                t = state["t"]
                m: torch.Tensor = state["m"]
                v: torch.Tensor = state["v"]    

                # update moments and learning rate for this iteration t in-place
                m.mul_(beta1).add_(grad, alpha= 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value= 1 - beta2)

                lr_t = lr * (math.sqrt(1 - beta2**t)) / (1 - beta1**t)

                # update params and apply weight decay in place
                p.data.addcdiv_(m, torch.sqrt(v) + epsilon, value=-lr_t) # p.data = p.data + (-lr_t * m / v.sqrt + eps)
                p.data.mul_(1 - lr * weight_decay)  # p..data = p.data - lr*wd*p.data

                # update state variables
                state["t"] = t + 1

        return loss



    
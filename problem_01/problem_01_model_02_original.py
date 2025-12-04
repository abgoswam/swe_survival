import torch
import torch.nn as nn


class WeightedSumEnsemble(nn.Module):
    """A module that combines a list of submodules by a trainable weighted sum.

    Args:
        modules (list[nn.Module]): list of modules that each accept the same input
            tensor `x` with shape (B, E1) and produce outputs of shape (B, E2).
        normalize (bool): if True, weights are normalized with a softmax before
            combining. If False, raw scalar weights are used.

    Forward:
        Given input x, compute outputs_i = modules[i](x), then return
        sum_i w_i * outputs_i (weights broadcast over batch and features).
    """

    def __init__(self, modules, normalize: bool = True):
        super().__init__()
        if not isinstance(modules, (list, tuple)):
            raise TypeError("modules must be a list/tuple of nn.Modules")
        if len(modules) == 0:
            raise ValueError("modules must contain at least one submodule")

        # Store submodules as a ModuleList so parameters are registered
        self.modules_list = nn.ModuleList(modules)
        self.n = len(modules)
        self.normalize = normalize

        # Initialize weights evenly so the initial output is the simple mean.
        init = torch.ones(self.n, dtype=torch.float32) / float(self.n)
        self.weights = nn.Parameter(init)

    def forward(self, x):
        # Collect outputs
        outs = [m(x) for m in self.modules_list]
        if len(outs) == 0:
            raise RuntimeError("no outputs produced by submodules")

        # Ensure all outputs have the same shape
        base_shape = outs[0].shape
        for idx, t in enumerate(outs):
            if t.shape != base_shape:
                raise ValueError(
                    f"output shape mismatch at index {idx}: {t.shape} != {base_shape}"
                )

        if self.normalize:
            w = torch.softmax(self.weights, dim=0)
        else:
            w = self.weights

        # Compute weighted sum: shape (n,) -> (n, 1, 1, ... ) to broadcast properly
        # We'll unsqueeze for batch and feature dims. For 2D outputs (B,E) we need (n,1,1)
        # but simpler: multiply each output by scalar and sum.
        out = None
        for i, t in enumerate(outs):
            term = w[i] * t
            out = term if out is None else out + term

        return out


__all__ = ["WeightedSumEnsemble"]

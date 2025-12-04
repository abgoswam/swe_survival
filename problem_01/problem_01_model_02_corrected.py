import torch
import torch.nn as nn


class WeightedSumEnsemble(nn.Module):
    """A module that combines a list of submodules by a trainable weighted sum.

    Args:
        modules (list[nn.Module]): a list or tuple of submodules, each of which
            accepts the same input tensor `x` with shape (B, E1) and produces
            outputs of shape (B, E2). The constructor requires a single
            list/tuple argument and does not accept varargs.

    Forward:
        Given input x, compute outputs_i = modules[i](x), then return
        sum_i w_i * outputs_i (weights broadcast over batch and features).
    """

    def __init__(self, modules):
        super().__init__()
        if not isinstance(modules, (list, tuple)):
            raise TypeError("modules must be provided as a list or tuple of nn.Modules")
        if len(modules) == 0:
            raise ValueError("modules must contain at least one submodule")

        # Store submodules as a ModuleList so parameters are registered
        self.modules_list = nn.ModuleList(modules)
        self.n = len(self.modules_list)
        # Initialize raw scalar trainable weights randomly (positive) and
        # normalize so they sum to 1. This breaks symmetry between identical
        # submodules while keeping the ensemble output scale similar to the
        # simple mean at initialization.
        init = torch.rand(self.n, dtype=torch.float32)
        init = init / init.sum()
        self.weights = nn.Parameter(init)

    def forward(self, x):
        # Accumulate outputs on-the-fly to avoid storing all outputs at once.
        # This reduces peak temporary memory from O(n * B * E) to O(B * E).
        w = self.weights
        out = None
        base_shape = None
        for i, m in enumerate(self.modules_list):
            t = m(x)
            if out is None:
                # first output: set base shape and initialize accumulator
                if t is None:
                    raise RuntimeError("no outputs produced by submodules")
                base_shape = t.shape
                out = w[i] * t
                continue

            # shape check for subsequent outputs
            if t.shape != base_shape:
                raise ValueError(
                    f"output shape mismatch at index {i}: {t.shape} != {base_shape}"
                )

            # accumulate
            out = out + w[i] * t

        return out


# Leave public API to normal import conventions - no __all__ exported here.

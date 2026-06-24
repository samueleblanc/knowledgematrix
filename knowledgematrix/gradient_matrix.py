import math
import torch
from torch import nn
from typing import Union

from knowledgematrix.neural_net import NN


# Layer types whose composition is piecewise-linear in the input.
_PL_LAYER_TYPES = (
    nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.Flatten,
    nn.ReLU, nn.LeakyReLU,
    nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d,
    nn.BatchNorm2d, nn.Dropout, nn.Identity,
)


def _assert_piecewise_linear(model: NN) -> None:
    """
        Raises ValueError if any layer of the model is not piecewise-linear.

        Args:
            model (NN): The network to check.
    """
    for i, layer in enumerate(model.layers):
        if not isinstance(layer, _PL_LAYER_TYPES):
            raise ValueError(
                f"Layer {i} ({type(layer).__name__}) is not piecewise-linear. "
                f"GradientMatrixComputer is exact only for piecewise-linear networks "
                f"(ReLU-family activations, affine layers, max/avg pooling). "
                f"Use KnowledgeMatrixComputer for this model."
            )


class GradientMatrixComputer:
    """
        Computes the knowledge matrix of a piecewise-linear neural network as
        per-class gradient x input, with a FullGrad bias column.

        Args:
            model (NN): The network. Must be piecewise-linear and in eval mode.
            batch_size (int): Output classes per chunk for the autograd backend.
            device (Union[str, None]): Compute device; defaults to model.device.
            backend (str): "func" (torch.func.jacrev, default) or "autograd"
                (chunked torch.autograd.grad) for the input Jacobian.
    """

    def __init__(
            self,
            model: NN,
            batch_size: int = 1,
            device: Union[str, None] = None,
            backend: str = "func"
        ) -> None:
        if backend not in ("func", "autograd"):
            raise ValueError(f"backend must be 'func' or 'autograd', got {backend!r}")
        _assert_piecewise_linear(model)
        self.model = model
        self.batch_size = batch_size
        self.device = device if device is not None else model.device
        self.backend = backend
        self.input_shape = model.input_shape
        self.input_size = math.prod(model.input_shape)
        self.current_output: Union[torch.Tensor, None] = None

    def forward(self, x: torch.Tensor, extract_weff: bool = False) -> torch.Tensor:
        """
            Computes the knowledge matrix at a given input point.

            Args:
                x (torch.Tensor): Input of shape model.input_shape (unbatched).
                extract_weff (bool): If True, return the input Jacobian
                    W_eff[c, i] = df_c/dx_i of shape (output_size, input_size).
                    If False (default), return gradient x input with an appended
                    bias column of shape (output_size, input_size + 1).
            Returns:
                torch.Tensor: The knowledge matrix (default) or W_eff.
        """
        # NN.eval()/train() set per-layer modes but do not flip the container's
        # own `training` flag, so inspect the actual layers (the true source of
        # BatchNorm/Dropout behaviour) rather than self.model.training.
        if any(layer.training for layer in self.model.layers):
            raise RuntimeError(
                "Call model.eval() before computing the gradient knowledge matrix "
                "(BatchNorm/Dropout must use eval-mode behaviour)."
            )
        self.model.to(self.device)
        x = x.to(self.device).reshape(self.input_shape)
        J = self._jacobian(x)
        self.current_output = self._forward_output(x)
        if extract_weff:
            return J
        raise NotImplementedError  # bias column added in Task 3

    def _forward_output(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.forward(x).flatten()

    def _jacobian(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "func":
            return self._jacobian_func(x)
        return self._jacobian_autograd(x)

    def _jacobian_func(self, x: torch.Tensor) -> torch.Tensor:
        from torch.func import jacrev

        def f(x_flat: torch.Tensor) -> torch.Tensor:
            return self.model.forward(x_flat.reshape(self.input_shape)).flatten()

        return jacrev(f)(x.flatten())

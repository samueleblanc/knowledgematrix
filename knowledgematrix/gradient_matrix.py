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

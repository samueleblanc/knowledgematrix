"""
    Implementation of ResNet-152 (He et al., 2015) — Bottleneck variant.

    Notes for the KM framework
    --------------------------
    ResNet-152 uses ``Bottleneck`` blocks with three convolutions
    (1x1 -> 3x3 -> 1x1) and three ReLU calls per block (after bn1, after
    bn2, and after the residual add). Unlike the existing ``resnet18.py``
    wrapper, this file places ReLUs at every semantic position so that the
    KM-side forward matches torchvision's ``resnet152`` exactly. The library's
    ``NN.residual(start, end)`` method does NOT inject a post-residual ReLU
    on its own (see ``NN.apply_residual`` in ``neural_net.py``: it only does
    ``x = x + projection(identity)``), so the post-residual ReLU must be
    appended explicitly. The convention used below is to register the
    residual against the NEXT layer's index and let that next layer be the
    post-residual ReLU — the framework applies ``apply_residual`` BEFORE the
    layer at ``end`` runs, so this places the ReLU at exactly the right spot.

    BatchNorm running-stats hazard
    ------------------------------
    ``NN.residual(start, end)`` internally calls ``shape_at_layer(start)``
    and ``shape_at_layer(end)`` to decide on a projection shape. Those calls
    run a fake forward pass through every previously-appended layer using
    random input. While the layers are in training mode (the default after
    ``nn.BatchNorm2d`` construction), every such call mutates each BN's
    ``running_mean`` / ``running_var`` toward the random-input statistics. By
    the time the model is fully built, the pretrained BN running stats are
    completely corrupted (and the corruption is shared with the source
    ``pretrained_model`` because modules are aliased, not copied). To avoid
    this, every BatchNorm2d module is put into ``eval()`` mode immediately
    upon being appended to ``self.layers`` and likewise for residual-
    projection BNs after each ``residual()`` call. The user's ``NN.eval()``
    later in their pipeline is then a no-op for these BNs, which is the
    desired behavior.
"""
from typing import Any, Union

from torch import nn
from torchvision.models import resnet152, ResNet152_Weights

from knowledgematrix.neural_net import NN


# (num_blocks, planes, stride_first_block)
# planes is the "inner" channel count; the block emits planes * 4 channels.
_RESNET152_LAYERS = (
    (3, 64, 1),    # layer1: 3 Bottlenecks, no spatial downsample, expand 64 -> 256
    (8, 128, 2),   # layer2: 8 Bottlenecks, stride-2 in first block, 256 -> 512
    (36, 256, 2),  # layer3: 36 Bottlenecks, stride-2 in first block, 512 -> 1024
    (3, 512, 2),   # layer4: 3 Bottlenecks, stride-2 in first block, 1024 -> 2048
)
_BOTTLENECK_EXPANSION = 4


class ResNet152(NN):
    """
        The ResNet-152 model (Bottleneck variant).

        Args:
            input_shape (tuple[int]): The shape of the input to the network.
            num_classes (int): The number of classes in the dataset.
            save (bool): Whether to save the activations and preactivations of the network.
            pretrained (bool): Whether to use pretrained weights.
            pretrained_model (Union[torch.nn.Module, None]): You can pass in a custom
                ResNet-152 module to use as pretrained weights. If None, the default
                pretrained weights will be used.
            device (str): The device to run the network on.
    """
    def __init__(
            self,
            input_shape: tuple[int],
            num_classes: int,
            save: bool = False,
            pretrained: bool = False,
            pretrained_model: Union[Any, None] = None,
            device: str = "cpu",
        ) -> None:
        super().__init__(input_shape, save, device)

        if pretrained:
            if pretrained_model is None:
                pretrained_model = resnet152(weights=ResNet152_Weights.DEFAULT)
            self._build_from_pretrained(pretrained_model)
            if input_shape[0] != 3:
                print(
                    f"Warning: The pretrained model was trained on 3-channel images. "
                    f"The input shape is {input_shape}. The first layer won't have pretrained weights."
                )
                self.layers[0] = nn.Conv2d(
                    input_shape[0], 64,
                    kernel_size=7, stride=2, padding=3, bias=False,
                )
            elif num_classes != 1000:
                print(
                    f"Warning: The pretrained model was trained on 1000 classes. "
                    f"The number of classes is {num_classes}. The last layer won't have pretrained weights."
                )
                self.layers[-1] = nn.Linear(2048, num_classes)
        else:
            self._build_custom(input_shape[0], num_classes)

    # ------------------------------------------------------------------ #
    # Pretrained construction (introspect torchvision's resnet152)        #
    # ------------------------------------------------------------------ #
    def _build_from_pretrained(self, pretrained_model: nn.Module) -> None:
        for name, module in pretrained_model.named_children():
            if name == "conv1":
                self.layers.append(module)
            elif name == "bn1":
                self._append_bn_eval(module)
            elif name == "relu":
                # Stem ReLU (post bn1).
                self.relu()
            elif name == "maxpool":
                self.maxpool(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                )
            elif name in ("layer1", "layer2", "layer3", "layer4"):
                for bottleneck in module.children():
                    self._append_pretrained_bottleneck(bottleneck)
            elif name == "avgpool":
                self.adaptiveavgpool(output_size=module.output_size)
                self.flatten()
            elif name == "fc":
                self.layers.append(module)

    def _append_pretrained_bottleneck(self, bottleneck: nn.Module) -> None:
        """Append a single Bottleneck block's modules using its torchvision instance."""
        start_skip = self.get_num_layers()
        # 1x1 conv -> bn -> relu
        self.layers.append(bottleneck.conv1)
        self._append_bn_eval(bottleneck.bn1)
        self.relu()
        # 3x3 conv (may carry stride for downsampling block) -> bn -> relu
        self.layers.append(bottleneck.conv2)
        self._append_bn_eval(bottleneck.bn2)
        self.relu()
        # 1x1 expansion conv -> bn (no relu yet — residual add comes first)
        self.layers.append(bottleneck.conv3)
        self._append_bn_eval(bottleneck.bn3)
        # Register the residual to fire immediately BEFORE the next layer (the
        # post-residual ReLU). Whether the auto-projection or the actual
        # downsample sequential is used, the residual end is the same index.
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        if bottleneck.downsample is not None:
            # Override the auto-generated projection with torchvision's actual
            # downsample modules so the pretrained weights are used.
            for i, downsample_layer in enumerate(bottleneck.downsample.children()):
                self.residuals[end_skip][0][1][i] = downsample_layer
        # Freeze running stats on every BN inside the residual projection so
        # subsequent shape_at_layer() calls don't corrupt them.
        for _, proj in self.residuals[end_skip]:
            for sub in proj:
                if isinstance(sub, nn.BatchNorm2d):
                    sub.eval()
        # Post-residual ReLU (this layer index == end_skip; the residual is
        # added to x BEFORE this layer runs).
        self.relu()

    def _append_bn_eval(self, bn_module: nn.BatchNorm2d) -> None:
        """
            Append a BatchNorm2d module already in eval() mode. This protects
            its ``running_mean`` / ``running_var`` from being silently mutated
            by the random-input forward inside ``NN.shape_at_layer``, which is
            invoked twice per ``self.residual(...)`` call.
        """
        bn_module.eval()
        self.layers.append(bn_module)

    # ------------------------------------------------------------------ #
    # Custom construction (no pretrained weights)                         #
    # ------------------------------------------------------------------ #
    def _build_custom(self, in_channels: int, num_classes: int) -> None:
        # Stem: 7x7 conv, stride 2, padding 3 -> bn -> relu -> 3x3 maxpool stride 2.
        self.conv(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self._batchnorm_eval(64)
        self.relu()
        self.maxpool(kernel_size=3, stride=2, padding=1)

        cur_in = 64
        for num_blocks, planes, stride_first in _RESNET152_LAYERS:
            for b in range(num_blocks):
                stride = stride_first if b == 0 else 1
                self._add_custom_bottleneck(cur_in, planes, stride)
                cur_in = planes * _BOTTLENECK_EXPANSION

        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.linear(in_features=2048, out_features=num_classes)

    def _add_custom_bottleneck(self, in_planes: int, planes: int, stride: int) -> None:
        """
            Add a single Bottleneck block. Stride may be > 1 only for the first
            block of each ResNet stage; the 3x3 conv carries the stride and the
            framework's residual auto-projection handles the identity branch.
        """
        out_planes = planes * _BOTTLENECK_EXPANSION
        start_skip = self.get_num_layers()
        # 1x1 reduce
        self.conv(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self._batchnorm_eval(planes)
        self.relu()
        # 3x3 (carries stride if any)
        self.conv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self._batchnorm_eval(planes)
        self.relu()
        # 1x1 expand
        self.conv(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self._batchnorm_eval(out_planes)
        # Residual: framework auto-creates a 1x1 conv + bn projection when shapes
        # differ (stride>1 OR in_planes != out_planes), and Identity otherwise.
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        for _, proj in self.residuals[end_skip]:
            for sub in proj:
                if isinstance(sub, nn.BatchNorm2d):
                    sub.eval()
        # Post-residual ReLU.
        self.relu()

    def _batchnorm_eval(self, num_features: int) -> None:
        """
            Append a fresh BatchNorm2d already in eval() mode. See the
            module-level docstring's "BatchNorm running-stats hazard" note.
        """
        self.batchnorm(num_features)
        self.layers[-1].eval()

"""
    Implementation of ResNet-18 (He et al., 2015) — BasicBlock variant.

    Notes for the KM framework
    --------------------------
    ResNet-18 uses ``BasicBlock`` blocks with two 3x3 convolutions and TWO
    ReLU calls per block: one after ``bn1`` and one AFTER the residual add.
    The previous version of this wrapper omitted the post-residual-add ReLU
    (it emitted only 9 ReLUs vs torchvision's 17 call-sites), so its forward
    pass did NOT match torchvision's ``resnet18`` even though the KM invariant
    (``mat.sum(1) == model.forward(x)``) still held against its own wrong
    forward. The fix follows the convention already used in ``resnet152.py``:

      * ``NN.residual(start, end)`` does NOT inject a post-residual ReLU on
        its own (see ``NN.apply_residual`` in ``neural_net.py``: it only does
        ``x = x + projection(identity)``). The framework applies
        ``apply_residual`` BEFORE the layer at index ``end`` runs, so the
        residual is registered against the index of the post-add ReLU and
        that ReLU is made the layer at ``end``. This places
        ``relu(out + identity)`` at exactly the spot torchvision uses.

    BatchNorm running-stats hazard
    ------------------------------
    ``NN.residual(start, end)`` calls ``shape_at_layer`` twice, each running a
    random-input forward through every previously-appended layer. While the
    BatchNorms are in training mode (the default after construction), those
    passes mutate their ``running_mean`` / ``running_var`` -- corrupting the
    pretrained statistics (and, since modules are aliased not copied, the
    statistics of the source ``pretrained_model`` too) -- and on small inputs
    that collapse to 1x1 spatial they even raise "Expected more than 1 value
    per channel when training". Every BatchNorm2d (layer BNs and residual-
    projection BNs) is therefore put into ``eval()`` immediately upon
    construction. The user's later ``NN.eval()`` is then a no-op for these
    BNs, which is the desired behavior. See ``resnet152.py`` for the original
    write-up of this hazard.
"""
from typing import Any, Union

from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.resnet import ResNet

from knowledgematrix.neural_net import NN


# (num_blocks, planes, stride_first_block) for the BasicBlock variant.
# BasicBlock has expansion 1, so a block emits ``planes`` channels.
_RESNET18_LAYERS = (
    (2, 64, 1),    # layer1: 2 BasicBlocks, no spatial downsample
    (2, 128, 2),   # layer2: stride-2 in first block, 64 -> 128
    (2, 256, 2),   # layer3: stride-2 in first block, 128 -> 256
    (2, 512, 2),   # layer4: stride-2 in first block, 256 -> 512
)
_BASICBLOCK_EXPANSION = 1


class ResNet18(NN):
    """
        The ResNet-18 model (BasicBlock variant).

        Args:
            input_shape (tuple[int]): The shape of the input to the network.
            num_classes (int): The number of classes in the dataset.
            save (bool): Whether to save the activations and preactivations of the network.
            pretrained (bool): Whether to use pretrained weights.
            pretrained_model (Union[torch.nn.Module, None]): You can pass in a custom
                ResNet-18 module to use as pretrained weights. If None, the default
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
                pretrained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
            elif not isinstance(pretrained_model, ResNet):
                # NOTE: the old check used the torchvision *function* ``resnet18``
                # as the isinstance type argument, which always raised TypeError.
                # The correct type is the ``ResNet`` class returned by the builder.
                raise ValueError(
                    "pretrained_model must be an instance of torchvision.models.resnet.ResNet."
                )
            self._build_from_pretrained(pretrained_model)
            if input_shape[0] != 3:
                print(
                    f"Warning: The pretrained model was trained on 3-channel images. "
                    f"The input shape is {input_shape}. The first layer won't have pretrained weights."
                )
                self.layers[0] = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
            elif num_classes != 1000:
                print(
                    f"Warning: The pretrained model was trained on 1000 classes. "
                    f"The number of classes is {num_classes}. The last layer won't have pretrained weights."
                )
                self.layers[-1] = nn.Linear(512, num_classes)
        else:
            self._build_custom(input_shape[0], num_classes)

    # ------------------------------------------------------------------ #
    # Pretrained construction (introspect torchvision's resnet18)         #
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
                for block in module.children():
                    self._append_pretrained_basicblock(block)
            elif name == "avgpool":
                self.adaptiveavgpool(output_size=module.output_size)
                self.flatten()
            elif name == "fc":
                self.layers.append(module)

    def _append_pretrained_basicblock(self, block: nn.Module) -> None:
        """Append a single BasicBlock's modules using its torchvision instance."""
        start_skip = self.get_num_layers()
        # 3x3 conv (may carry stride for a downsampling block) -> bn -> relu
        self.layers.append(block.conv1)
        self._append_bn_eval(block.bn1)
        self.relu()
        # 3x3 conv -> bn (no relu yet — the residual add comes first)
        self.layers.append(block.conv2)
        self._append_bn_eval(block.bn2)
        # Register the residual to fire immediately BEFORE the next layer (the
        # post-residual ReLU). The end index is the same whether the auto
        # projection or the real downsample sequential is used.
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        if block.downsample is not None:
            # Override the auto-generated projection with torchvision's actual
            # downsample modules so the pretrained weights are used.
            for i, downsample_layer in enumerate(block.downsample.children()):
                self.residuals[end_skip][0][1][i] = downsample_layer
        self._freeze_projection_bns(end_skip)
        # Post-residual ReLU (layer index == end_skip; the residual is added
        # to x BEFORE this layer runs).
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
        # Stem (kept identical to the original wrapper): 3x3 conv stride 1 ->
        # bn -> relu -> 3x3 maxpool stride 2.
        self.conv(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self._batchnorm_eval(64)
        self.relu()
        self.maxpool(kernel_size=3, stride=2, padding=1)

        cur_in = 64
        for num_blocks, planes, stride_first in _RESNET18_LAYERS:
            for b in range(num_blocks):
                stride = stride_first if b == 0 else 1
                self._add_custom_basicblock(cur_in, planes, stride)
                cur_in = planes * _BASICBLOCK_EXPANSION

        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.linear(in_features=512, out_features=num_classes)

    def _add_custom_basicblock(self, in_planes: int, planes: int, stride: int) -> None:
        """
            Add a single BasicBlock. Stride may be > 1 only for the first block
            of each ResNet stage; the first 3x3 conv carries the stride and the
            framework's residual auto-projection handles the identity branch.
        """
        out_planes = planes * _BASICBLOCK_EXPANSION
        start_skip = self.get_num_layers()
        # 3x3 (carries stride if any) -> bn -> relu
        self.conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self._batchnorm_eval(planes)
        self.relu()
        # 3x3 -> bn (no relu yet — residual add comes first)
        self.conv(planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self._batchnorm_eval(out_planes)
        # Residual: framework auto-creates a 1x1 conv + bn projection when shapes
        # differ (stride>1 OR in_planes != out_planes), and Identity otherwise.
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self._freeze_projection_bns(end_skip)
        # Post-residual ReLU.
        self.relu()

    def _batchnorm_eval(self, num_features: int) -> None:
        """
            Append a fresh BatchNorm2d already in eval() mode. See the
            module-level docstring's "BatchNorm running-stats hazard" note.
        """
        self.batchnorm(num_features)
        self.layers[-1].eval()

    def _freeze_projection_bns(self, end_skip: int) -> None:
        """
            Freeze running stats on every BN inside a residual projection so
            subsequent ``shape_at_layer`` calls (fired by later residuals)
            don't corrupt them.
        """
        for _, proj in self.residuals[end_skip]:
            for sub in proj:
                if isinstance(sub, nn.BatchNorm2d):
                    sub.eval()

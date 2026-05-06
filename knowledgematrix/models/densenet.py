"""
    Implementation of DenseNet (Huang et al., 2017).

    Supports torchvision-style pretrained variants (densenet121/161/169/201)
    and custom DenseNets parameterized by `growth_rate`, `block_config`,
    `num_init_features`, `bn_size`, `drop_rate`, and `compression`.
"""
from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models.densenet import DenseNet as _TVDenseNet
from typing import Tuple, Union

from knowledgematrix.neural_net import NN


class DenseNet(NN):
    """
        DenseNet model with channel-wise concatenation skip connections.

        Args:
            input_shape (Tuple[int]): The shape of the input to the network.
            num_classes (int): The number of classes in the dataset.
            save (bool): Whether to save the activations and preactivations of the network.
            pretrained (bool): Whether to use pretrained weights. Loads densenet121 weights by default.
            pretrained_model (Union[_TVDenseNet, None]): Pass a torchvision DenseNet instance
                (e.g. `densenet121(...)`, `densenet161(...)`) to use as the source of pretrained
                weights and architecture. If None and `pretrained` is True, the default
                densenet121 with DEFAULT weights is loaded.
            device (str): The device to run the network on.
            growth_rate (int): How many features each dense layer adds (k in the paper).
                Ignored when `pretrained=True`.
            block_config (Tuple[int, ...]): Number of dense layers per dense block.
                Ignored when `pretrained=True`.
            num_init_features (int): Number of features after the stem conv.
                Ignored when `pretrained=True`.
            bn_size (int): Bottleneck multiplier (1x1 conv outputs `bn_size * growth_rate` channels).
                Ignored when `pretrained=True`.
            drop_rate (float): Dropout rate after each dense layer's 3x3 conv. 0 disables dropout.
            compression (float): Channel compression factor in transition layers.
                Ignored when `pretrained=True` (torchvision hardcodes 0.5 via `num_features // 2`).
    """
    def __init__(
            self,
            input_shape: Tuple[int],
            num_classes: int,
            save: bool = False,
            pretrained: bool = False,
            pretrained_model: Union[_TVDenseNet, None] = None,
            device: str = "cpu",
            growth_rate: int = 32,
            block_config: Tuple[int, ...] = (6, 12, 24, 16),
            num_init_features: int = 64,
            bn_size: int = 4,
            drop_rate: float = 0.0,
            compression: float = 0.5,
        ) -> None:
        super().__init__(input_shape, save, device)

        if pretrained:
            if pretrained_model is None:
                pretrained_model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            else:
                if not isinstance(pretrained_model, _TVDenseNet):
                    raise ValueError("pretrained_model must be an instance of torchvision DenseNet.")
            self._build_from_pretrained(pretrained_model, drop_rate)
            init_conv_out = self.layers[0].out_channels
            classifier_in = self.layers[-1].in_features
            if input_shape[0] != 3:
                print(f"Warning: The pretrained model was trained on 3-channel images. The input shape is {input_shape}. The first layer won't have pretrained weights.")
                self.layers[0] = nn.Conv2d(input_shape[0], init_conv_out, kernel_size=7, stride=2, padding=3, bias=False)
            elif num_classes != 1000:
                print(f"Warning: The pretrained model was trained on 1000 classes. The number of classes is {num_classes}. The last layer won't have pretrained weights.")
                self.layers[-1] = nn.Linear(classifier_in, num_classes)
        else:
            self._build_custom(
                in_channels=input_shape[0],
                num_classes=num_classes,
                growth_rate=growth_rate,
                block_config=block_config,
                num_init_features=num_init_features,
                bn_size=bn_size,
                drop_rate=drop_rate,
                compression=compression,
            )

    def _build_custom(
            self,
            in_channels: int,
            num_classes: int,
            growth_rate: int,
            block_config: Tuple[int, ...],
            num_init_features: int,
            bn_size: int,
            drop_rate: float,
            compression: float,
        ) -> None:
        # Stem
        self.conv(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchnorm(num_init_features)
        self.relu()
        self.maxpool(kernel_size=3, stride=2, padding=1)

        n_feat = num_init_features
        for bi, n_layers in enumerate(block_config):
            block_starts = [self.get_num_layers()]
            for li in range(n_layers):
                cur_in = n_feat + li * growth_rate
                self.batchnorm(cur_in)
                self.relu()
                self.conv(cur_in, bn_size * growth_rate, kernel_size=1, bias=False)
                self.batchnorm(bn_size * growth_rate)
                self.relu()
                self.conv(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
                if drop_rate > 0:
                    self.dropout(p=drop_rate)
                block_starts.append(self.get_num_layers())
            # Wire concat skips: dest at block_starts[k] receives sources block_starts[0..k-1]
            for k in range(1, len(block_starts)):
                for s in range(k):
                    self.concat_skip(block_starts[s], block_starts[k])
            n_feat = n_feat + n_layers * growth_rate
            # Transition between blocks (not after the last block)
            if bi != len(block_config) - 1:
                out_feat = int(n_feat * compression)
                self.batchnorm(n_feat)
                self.relu()
                self.conv(n_feat, out_feat, kernel_size=1, bias=False)
                self.avgpool(kernel_size=2, stride=2)
                n_feat = out_feat

        # Final BN + ReLU + AdaptiveAvgPool + Flatten + Linear
        self.batchnorm(n_feat)
        self.relu()
        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.linear(n_feat, num_classes)

    def _build_from_pretrained(self, pretrained_model: _TVDenseNet, drop_rate: float) -> None:
        features = pretrained_model.features
        for name, module in features.named_children():
            if name == "conv0":
                self.layers.append(module)
            elif name == "norm0":
                self.layers.append(module)
            elif name == "relu0":
                self.relu()
            elif name == "pool0":
                self.maxpool(
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                )
            elif name.startswith("denseblock"):
                block_starts = [self.get_num_layers()]
                for _, dl in module.named_children():
                    self.layers.append(dl.norm1)
                    self.relu()
                    self.layers.append(dl.conv1)
                    self.layers.append(dl.norm2)
                    self.relu()
                    self.layers.append(dl.conv2)
                    if drop_rate > 0:
                        self.dropout(p=drop_rate)
                    block_starts.append(self.get_num_layers())
                for k in range(1, len(block_starts)):
                    for s in range(k):
                        self.concat_skip(block_starts[s], block_starts[k])
            elif name.startswith("transition"):
                for _, sub in module.named_children():
                    if isinstance(sub, nn.BatchNorm2d):
                        self.layers.append(sub)
                    elif isinstance(sub, nn.ReLU):
                        self.relu()
                    elif isinstance(sub, nn.Conv2d):
                        self.layers.append(sub)
                    elif isinstance(sub, nn.AvgPool2d):
                        self.avgpool(
                            kernel_size=sub.kernel_size,
                            stride=sub.stride,
                            padding=sub.padding,
                        )
            elif name == "norm5":
                self.layers.append(module)
        # Final activation pipeline lives in torchvision's DenseNet.forward, not in features
        self.relu()
        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.layers.append(pretrained_model.classifier)

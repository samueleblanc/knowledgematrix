"""
    Implementation of Inception-v3 (Szegedy et al., 2016).

    Supports the torchvision pretrained inception_v3 (auxiliary classifier
    head disabled at eval) and fully custom Inception-v3-style networks
    parameterized by a channel scale factor.

    Linearizing parallel branches: each Inception module forks the module
    input across N branches that are later concatenated. We linearize this
    via NN.branch_input(start, end) -- after each branch finishes, the next
    branch starts by restoring x to the snapshot of the module's fork
    point. The N-1 intermediate branch outputs are captured at branch
    boundaries (which double as concat_skip sources for the next module's
    first layer); the last branch's output naturally is "current x" when
    the next module starts. concat_skip then assembles the N-way merge at
    the next module's first layer.
"""
from torch import nn
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.inception import (
    Inception3 as _TVInception3,
    BasicConv2d as _TVBasicConv2d,
    InceptionA as _TVInceptionA,
    InceptionB as _TVInceptionB,
    InceptionC as _TVInceptionC,
    InceptionD as _TVInceptionD,
    InceptionE as _TVInceptionE,
)
from typing import Tuple, Union

from knowledgematrix.neural_net import NN


def _round8(n: int) -> int:
    return max(8, int(round(n / 8)) * 8)


class Inception(NN):
    """
        Inception-v3 model with linearized parallel branches.

        Args:
            input_shape (Tuple[int]): Shape of the input to the network (C, H, W).
                For pretrained variants, H, W >= 75 is recommended (architecture
                minimum).
            num_classes (int): Number of output classes.
            save (bool): Whether to save activations and preactivations.
            pretrained (bool): If True, load torchvision inception_v3 weights.
            pretrained_model (Union[_TVInception3, None]): Optional torchvision
                Inception3 instance to use as the source of weights and
                architecture. If None and `pretrained` is True, the default
                inception_v3 with DEFAULT weights (aux_logits=False) is used.
            device (str): The device to run the network on.
            scale (float): Channel-width multiplier used by the custom builder.
                Ignored when `pretrained=True`. All channel counts are scaled
                and rounded to the nearest multiple of 8 (with min 8).
    """
    def __init__(
            self,
            input_shape: Tuple[int],
            num_classes: int,
            save: bool = False,
            pretrained: bool = False,
            pretrained_model: Union[_TVInception3, None] = None,
            device: str = "cpu",
            scale: float = 1.0,
        ) -> None:
        super().__init__(input_shape, save, device)
        # Sources captured at branch boundaries of the most recently built
        # inception module. Wired into the next module's first layer (or the
        # final pooling layer) via concat_skip to produce the N-way channel
        # concatenation that is the module's output.
        self._pending_sources: list[int] = []

        if pretrained:
            if pretrained_model is None:
                pretrained_model = inception_v3(
                    weights=Inception_V3_Weights.DEFAULT,
                    aux_logits=True,
                )
            else:
                if not isinstance(pretrained_model, _TVInception3):
                    raise ValueError("pretrained_model must be an instance of torchvision Inception3.")
            pretrained_model.eval()
            self._build_from_pretrained(pretrained_model)
            stem_out = self.layers[0].out_channels
            classifier_in = self.layers[-1].in_features
            if input_shape[0] != 3:
                print(f"Warning: The pretrained model was trained on 3-channel images. The input shape is {input_shape}. The first layer won't have pretrained weights.")
                self.layers[0] = nn.Conv2d(input_shape[0], stem_out, kernel_size=3, stride=2, bias=False)
            elif num_classes != 1000:
                print(f"Warning: The pretrained model was trained on 1000 classes. The number of classes is {num_classes}. The last layer won't have pretrained weights.")
                self.layers[-1] = nn.Linear(classifier_in, num_classes)
        else:
            self._build_custom(input_shape[0], num_classes, scale)


    ### Helper methods ###

    def _basic_conv(
            self,
            in_ch: int,
            out_ch: int,
            kernel_size,
            stride=(1, 1),
            padding=(0, 0),
        ) -> None:
        # BasicConv2d in torchvision uses Conv2d (no bias) -> BatchNorm2d -> ReLU.
        # Match BN eps=0.001 to the torchvision default for inception_v3.
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.conv(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.batchnorm(out_ch, eps=0.001)
        self.relu()

    def _append_basic_conv(self, basic: _TVBasicConv2d) -> None:
        # Append a torchvision BasicConv2d (Conv2d + BatchNorm2d) plus a fresh
        # ReLU. BasicConv2d.forward applies F.relu, equivalent to nn.ReLU().
        self.layers.append(basic.conv)
        self.layers.append(basic.bn)
        self.relu()

    def _wire_pending_merge(self, target: int) -> None:
        """Wire the pending branch outputs into `target` via concat_skip,
        producing the prior module's N-way merge as the input at `target`."""
        for s in self._pending_sources:
            self.concat_skip(s, target)
        self._pending_sources = []


    ### Custom (parameterized) Inception-v3 builder ###

    def _build_custom(self, in_channels: int, num_classes: int, scale: float) -> None:
        c = lambda n: _round8(scale * n)

        # Stem
        self._basic_conv(in_channels, c(32), 3, stride=2)
        self._basic_conv(c(32), c(32), 3)
        self._basic_conv(c(32), c(64), 3, padding=1)
        self.maxpool(kernel_size=3, stride=2)
        self._basic_conv(c(64), c(80), 1)
        self._basic_conv(c(80), c(192), 3)
        self.maxpool(kernel_size=3, stride=2)

        in_ch = c(192)
        # Mixed_5b, 5c, 5d (InceptionA)
        in_ch = self._build_inception_a(in_ch, c(32))
        in_ch = self._build_inception_a(in_ch, c(64))
        in_ch = self._build_inception_a(in_ch, c(64))
        # Mixed_6a (InceptionB)
        in_ch = self._build_inception_b(in_ch)
        # Mixed_6b, 6c, 6d, 6e (InceptionC)
        in_ch = self._build_inception_c(in_ch, c(128))
        in_ch = self._build_inception_c(in_ch, c(160))
        in_ch = self._build_inception_c(in_ch, c(160))
        in_ch = self._build_inception_c(in_ch, c(192))
        # Mixed_7a (InceptionD)
        in_ch = self._build_inception_d(in_ch)
        # Mixed_7b, 7c (InceptionE)
        in_ch = self._build_inception_e(in_ch)
        in_ch = self._build_inception_e(in_ch)

        # Final head: AdaptiveAvgPool -> Flatten -> Linear (concat_skip merges
        # the last InceptionE's branches into the AdaptiveAvgPool's input)
        target = self.get_num_layers()
        self._wire_pending_merge(target)
        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.linear(in_ch, num_classes)


    ### Inception module builders (custom) ###

    def _build_inception_a(self, in_ch: int, pool_features: int) -> int:
        """InceptionA: 4 branches concatenated. Returns output channel count."""
        # Branches: branch1x1 (64ch), branch5x5 (48->64), branch3x3dbl (64->96->96),
        # branch_pool (avgpool then 1x1 to pool_features).
        c = lambda n: _round8(n)
        b1, b5, b3a, b3b = c(64), c(64), c(96), c(96)
        b_pool = pool_features

        fork = self.get_num_layers()
        self._wire_pending_merge(fork)

        # Branch 1
        self._basic_conv(in_ch, b1, 1)

        # Branch 2 (5x5)
        b2_start = self.get_num_layers()
        self.branch_input(fork, b2_start)
        self._basic_conv(in_ch, c(48), 1)
        self._basic_conv(c(48), b5, 5, padding=2)

        # Branch 3 (3x3 double)
        b3_start = self.get_num_layers()
        self.branch_input(fork, b3_start)
        self._basic_conv(in_ch, b3a, 1)
        self._basic_conv(b3a, b3b, 3, padding=1)
        self._basic_conv(b3b, b3b, 3, padding=1)

        # Branch 4 (avg pool -> 1x1)
        b4_start = self.get_num_layers()
        self.branch_input(fork, b4_start)
        self.avgpool(kernel_size=3, stride=1, padding=1)
        self._basic_conv(in_ch, b_pool, 1)

        self._pending_sources = [b2_start, b3_start, b4_start]
        return b1 + b5 + b3b + b_pool

    def _build_inception_b(self, in_ch: int) -> int:
        """InceptionB: 3 branches, downsampling stride-2."""
        c = lambda n: _round8(n)
        b3 = c(384)
        b3a, b3b, b3c = c(64), c(96), c(96)

        fork = self.get_num_layers()
        self._wire_pending_merge(fork)

        # Branch 1 (3x3 stride 2)
        self._basic_conv(in_ch, b3, 3, stride=2)

        # Branch 2 (3x3 double, last stride 2)
        b2_start = self.get_num_layers()
        self.branch_input(fork, b2_start)
        self._basic_conv(in_ch, b3a, 1)
        self._basic_conv(b3a, b3b, 3, padding=1)
        self._basic_conv(b3b, b3c, 3, stride=2)

        # Branch 3 (max pool 3x3 stride 2)
        b3_start = self.get_num_layers()
        self.branch_input(fork, b3_start)
        self.maxpool(kernel_size=3, stride=2)

        self._pending_sources = [b2_start, b3_start]
        return b3 + b3c + in_ch

    def _build_inception_c(self, in_ch: int, channels_7x7: int) -> int:
        """InceptionC: 4 branches with 1xn / nx1 factorized 7x7 convs."""
        c = lambda n: _round8(n)
        b1 = c(192)
        c7 = channels_7x7
        b_pool = c(192)

        fork = self.get_num_layers()
        self._wire_pending_merge(fork)

        # Branch 1
        self._basic_conv(in_ch, b1, 1)

        # Branch 2: 7x7 factorized as 1x1, 1x7, 7x1
        b2_start = self.get_num_layers()
        self.branch_input(fork, b2_start)
        self._basic_conv(in_ch, c7, 1)
        self._basic_conv(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self._basic_conv(c7, c(192), kernel_size=(7, 1), padding=(3, 0))

        # Branch 3: 7x7 double factorized
        b3_start = self.get_num_layers()
        self.branch_input(fork, b3_start)
        self._basic_conv(in_ch, c7, 1)
        self._basic_conv(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self._basic_conv(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self._basic_conv(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self._basic_conv(c7, c(192), kernel_size=(1, 7), padding=(0, 3))

        # Branch 4: avg pool -> 1x1
        b4_start = self.get_num_layers()
        self.branch_input(fork, b4_start)
        self.avgpool(kernel_size=3, stride=1, padding=1)
        self._basic_conv(in_ch, b_pool, 1)

        self._pending_sources = [b2_start, b3_start, b4_start]
        return b1 + c(192) + c(192) + b_pool

    def _build_inception_d(self, in_ch: int) -> int:
        """InceptionD: 3 branches, downsampling stride-2."""
        c = lambda n: _round8(n)

        fork = self.get_num_layers()
        self._wire_pending_merge(fork)

        # Branch 1: 1x1 -> 3x3 stride 2
        self._basic_conv(in_ch, c(192), 1)
        self._basic_conv(c(192), c(320), 3, stride=2)

        # Branch 2: 7x7 factorized then 3x3 stride 2
        b2_start = self.get_num_layers()
        self.branch_input(fork, b2_start)
        self._basic_conv(in_ch, c(192), 1)
        self._basic_conv(c(192), c(192), kernel_size=(1, 7), padding=(0, 3))
        self._basic_conv(c(192), c(192), kernel_size=(7, 1), padding=(3, 0))
        self._basic_conv(c(192), c(192), 3, stride=2)

        # Branch 3: max pool 3x3 stride 2
        b3_start = self.get_num_layers()
        self.branch_input(fork, b3_start)
        self.maxpool(kernel_size=3, stride=2)

        self._pending_sources = [b2_start, b3_start]
        return c(320) + c(192) + in_ch

    def _build_inception_e(self, in_ch: int) -> int:
        """InceptionE: 4 outer branches; outer branches 2 and 3 each split
        internally into two parallel sub-branches whose channel-wise concat
        is the outer branch's output. Channel concat is associative, so the
        next module's concat uses 5 explicit sources (branch1 + 4 sub-branch
        outputs from outer branches 2 & 3) plus current x = outer branch 4.
        No internal concat_skip is needed -- the next module reassembles
        everything in one merge."""
        c = lambda n: _round8(n)
        b1 = c(320)
        b3a, b3b = c(384), c(384)
        b3d_a, b3d_b, b3d_c = c(448), c(384), c(384)
        b_pool = c(192)

        fork = self.get_num_layers()
        self._wire_pending_merge(fork)

        # Branch 1: 1x1 -> 320
        self._basic_conv(in_ch, b1, 1)

        # Branch 2: 1x1 -> 384, then split into 1x3 (sub-a) || 3x1 (sub-b)
        b2_start = self.get_num_layers()
        self.branch_input(fork, b2_start)
        self._basic_conv(in_ch, b3a, 1)
        inner_fork_2 = self.get_num_layers()
        # Sub-branch a: 1x3
        self._basic_conv(b3a, b3b, kernel_size=(1, 3), padding=(0, 1))
        # Sub-branch b: 3x1
        sub2b_start = self.get_num_layers()
        self.branch_input(inner_fork_2, sub2b_start)
        self._basic_conv(b3a, b3b, kernel_size=(3, 1), padding=(1, 0))

        # Branch 3: 1x1 -> 448 -> 3x3 -> 384, then split into 1x3 (sub-a) || 3x1 (sub-b)
        b3_start = self.get_num_layers()
        self.branch_input(fork, b3_start)
        self._basic_conv(in_ch, b3d_a, 1)
        self._basic_conv(b3d_a, b3d_b, 3, padding=1)
        inner_fork_3 = self.get_num_layers()
        # Sub-branch a: 1x3
        self._basic_conv(b3d_b, b3d_c, kernel_size=(1, 3), padding=(0, 1))
        # Sub-branch b: 3x1
        sub3b_start = self.get_num_layers()
        self.branch_input(inner_fork_3, sub3b_start)
        self._basic_conv(b3d_b, b3d_c, kernel_size=(3, 1), padding=(1, 0))

        # Branch 4: avg pool -> 1x1
        b4_start = self.get_num_layers()
        self.branch_input(fork, b4_start)
        self.avgpool(kernel_size=3, stride=1, padding=1)
        self._basic_conv(in_ch, b_pool, 1)

        # Sources for the next module's concat (in cat-order). Each is a
        # pre-concat snapshot at the listed index:
        #   b2_start    -> branch 1 output
        #   sub2b_start -> branch 2 sub-a output
        #   b3_start    -> branch 2 sub-b output (sub2b's output flowing in)
        #   sub3b_start -> branch 3 sub-a output
        #   b4_start    -> branch 3 sub-b output
        # Plus current x at next module start = branch 4 output.
        self._pending_sources = [b2_start, sub2b_start, b3_start, sub3b_start, b4_start]
        return b1 + 2 * b3b + 2 * b3d_c + b_pool


    ### Pretrained (torchvision) Inception-v3 builder ###

    def _build_from_pretrained(self, pretrained_model: _TVInception3) -> None:
        # Stem
        self._append_basic_conv(pretrained_model.Conv2d_1a_3x3)
        self._append_basic_conv(pretrained_model.Conv2d_2a_3x3)
        self._append_basic_conv(pretrained_model.Conv2d_2b_3x3)
        self.maxpool(
            kernel_size=pretrained_model.maxpool1.kernel_size,
            stride=pretrained_model.maxpool1.stride,
            padding=pretrained_model.maxpool1.padding,
        )
        self._append_basic_conv(pretrained_model.Conv2d_3b_1x1)
        self._append_basic_conv(pretrained_model.Conv2d_4a_3x3)
        self.maxpool(
            kernel_size=pretrained_model.maxpool2.kernel_size,
            stride=pretrained_model.maxpool2.stride,
            padding=pretrained_model.maxpool2.padding,
        )

        # Mixed_5b, 5c, 5d (InceptionA)
        self._append_inception_a(pretrained_model.Mixed_5b)
        self._append_inception_a(pretrained_model.Mixed_5c)
        self._append_inception_a(pretrained_model.Mixed_5d)
        # Mixed_6a (InceptionB)
        self._append_inception_b(pretrained_model.Mixed_6a)
        # Mixed_6b, 6c, 6d, 6e (InceptionC)
        self._append_inception_c(pretrained_model.Mixed_6b)
        self._append_inception_c(pretrained_model.Mixed_6c)
        self._append_inception_c(pretrained_model.Mixed_6d)
        self._append_inception_c(pretrained_model.Mixed_6e)
        # Mixed_7a (InceptionD)
        self._append_inception_d(pretrained_model.Mixed_7a)
        # Mixed_7b, 7c (InceptionE)
        self._append_inception_e(pretrained_model.Mixed_7b)
        self._append_inception_e(pretrained_model.Mixed_7c)

        # Final head
        target = self.get_num_layers()
        self._wire_pending_merge(target)
        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.layers.append(pretrained_model.fc)


    ### Inception module appenders (pretrained) ###

    def _append_inception_a(self, m: _TVInceptionA) -> None:
        fork = self.get_num_layers()
        self._wire_pending_merge(fork)
        # Branch 1
        self._append_basic_conv(m.branch1x1)
        # Branch 2
        b2 = self.get_num_layers()
        self.branch_input(fork, b2)
        self._append_basic_conv(m.branch5x5_1)
        self._append_basic_conv(m.branch5x5_2)
        # Branch 3
        b3 = self.get_num_layers()
        self.branch_input(fork, b3)
        self._append_basic_conv(m.branch3x3dbl_1)
        self._append_basic_conv(m.branch3x3dbl_2)
        self._append_basic_conv(m.branch3x3dbl_3)
        # Branch 4 (avg pool -> 1x1)
        b4 = self.get_num_layers()
        self.branch_input(fork, b4)
        self.avgpool(kernel_size=3, stride=1, padding=1)
        self._append_basic_conv(m.branch_pool)
        self._pending_sources = [b2, b3, b4]

    def _append_inception_b(self, m: _TVInceptionB) -> None:
        fork = self.get_num_layers()
        self._wire_pending_merge(fork)
        # Branch 1: 3x3 stride 2
        self._append_basic_conv(m.branch3x3)
        # Branch 2: 1x1, 3x3, 3x3 stride 2
        b2 = self.get_num_layers()
        self.branch_input(fork, b2)
        self._append_basic_conv(m.branch3x3dbl_1)
        self._append_basic_conv(m.branch3x3dbl_2)
        self._append_basic_conv(m.branch3x3dbl_3)
        # Branch 3: max pool 3x3 stride 2
        b3 = self.get_num_layers()
        self.branch_input(fork, b3)
        self.maxpool(kernel_size=3, stride=2)
        self._pending_sources = [b2, b3]

    def _append_inception_c(self, m: _TVInceptionC) -> None:
        fork = self.get_num_layers()
        self._wire_pending_merge(fork)
        # Branch 1
        self._append_basic_conv(m.branch1x1)
        # Branch 2: 7x7 factorized
        b2 = self.get_num_layers()
        self.branch_input(fork, b2)
        self._append_basic_conv(m.branch7x7_1)
        self._append_basic_conv(m.branch7x7_2)
        self._append_basic_conv(m.branch7x7_3)
        # Branch 3: 7x7 double factorized
        b3 = self.get_num_layers()
        self.branch_input(fork, b3)
        self._append_basic_conv(m.branch7x7dbl_1)
        self._append_basic_conv(m.branch7x7dbl_2)
        self._append_basic_conv(m.branch7x7dbl_3)
        self._append_basic_conv(m.branch7x7dbl_4)
        self._append_basic_conv(m.branch7x7dbl_5)
        # Branch 4 (avg pool -> 1x1)
        b4 = self.get_num_layers()
        self.branch_input(fork, b4)
        self.avgpool(kernel_size=3, stride=1, padding=1)
        self._append_basic_conv(m.branch_pool)
        self._pending_sources = [b2, b3, b4]

    def _append_inception_d(self, m: _TVInceptionD) -> None:
        fork = self.get_num_layers()
        self._wire_pending_merge(fork)
        # Branch 1: 1x1 -> 3x3 stride 2
        self._append_basic_conv(m.branch3x3_1)
        self._append_basic_conv(m.branch3x3_2)
        # Branch 2: 7x7 factorized then 3x3 stride 2
        b2 = self.get_num_layers()
        self.branch_input(fork, b2)
        self._append_basic_conv(m.branch7x7x3_1)
        self._append_basic_conv(m.branch7x7x3_2)
        self._append_basic_conv(m.branch7x7x3_3)
        self._append_basic_conv(m.branch7x7x3_4)
        # Branch 3: max pool 3x3 stride 2
        b3 = self.get_num_layers()
        self.branch_input(fork, b3)
        self.maxpool(kernel_size=3, stride=2)
        self._pending_sources = [b2, b3]

    def _append_inception_e(self, m: _TVInceptionE) -> None:
        fork = self.get_num_layers()
        self._wire_pending_merge(fork)
        # Branch 1: 1x1 -> 320
        self._append_basic_conv(m.branch1x1)
        # Branch 2: 1x1 -> 384, then split into 1x3 (sub-a) || 3x1 (sub-b)
        b2 = self.get_num_layers()
        self.branch_input(fork, b2)
        self._append_basic_conv(m.branch3x3_1)
        inner_fork_2 = self.get_num_layers()
        self._append_basic_conv(m.branch3x3_2a)
        sub2b = self.get_num_layers()
        self.branch_input(inner_fork_2, sub2b)
        self._append_basic_conv(m.branch3x3_2b)
        # Branch 3: 1x1 -> 448 -> 3x3, then split into 1x3 (sub-a) || 3x1 (sub-b)
        b3 = self.get_num_layers()
        self.branch_input(fork, b3)
        self._append_basic_conv(m.branch3x3dbl_1)
        self._append_basic_conv(m.branch3x3dbl_2)
        inner_fork_3 = self.get_num_layers()
        self._append_basic_conv(m.branch3x3dbl_3a)
        sub3b = self.get_num_layers()
        self.branch_input(inner_fork_3, sub3b)
        self._append_basic_conv(m.branch3x3dbl_3b)
        # Branch 4 (avg pool -> 1x1)
        b4 = self.get_num_layers()
        self.branch_input(fork, b4)
        self.avgpool(kernel_size=3, stride=1, padding=1)
        self._append_basic_conv(m.branch_pool)
        # 5 explicit sources (in channel-cat order); branch 4 = current x at
        # the next module's start. Channel concat is associative, so the
        # next module's merge reproduces InceptionE's 4-way + nested-2-way
        # output without needing internal concat_skips.
        self._pending_sources = [b2, sub2b, b3, sub3b, b4]

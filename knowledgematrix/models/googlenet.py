"""
    Implementation of GoogLeNet (Inception v1) -- Szegedy et al., 2015,
    "Going Deeper with Convolutions" (arXiv:1409.4842), ILSVRC 2014 winner.

    Supports the torchvision pretrained ``googlenet`` (auxiliary classifier
    heads disabled at eval) and a fully custom GoogLeNet-style network.

    Architectural notes
    -------------------
    1. **No LRN.** The original 2014 paper used Local Response Normalization
       in the stem. Torchvision's ``googlenet`` implementation does NOT use
       LRN -- it uses BatchNorm2d (eps=0.001) inside ``BasicConv2d``. This
       wrapper follows torchvision exactly. No LRN substitution decision
       was needed; the BN-equipped torchvision variant IS the ground truth
       for the IMAGENET1K_V1 weights (which are ported from the original
       Caffe model and are designed to work with the torchvision BN
       structure). See:
       https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py

    2. **Auxiliary classifiers dropped.** The two auxiliary heads (``aux1``
       attached at inception4a, ``aux2`` attached at inception4d) are
       training-only regularizers (Szegedy et al. 2015 Sec. 5). With
       ``aux_logits=False`` torchvision sets them to ``None`` after loading
       weights, so they do not appear in the eval-mode forward and the
       pretrained main-path weights are unaffected.

    3. **Branch 3 uses 3x3 (not 5x5) convolutions.** This is a documented
       torchvision quirk -- a known mismatch with the paper:
       https://github.com/pytorch/vision/issues/906 .  We follow torchvision.

    4. **transform_input disabled.** torchvision's ``googlenet`` re-scales
       the input mean/std when ``transform_input=True`` (the default for
       the pretrained constructor). We force ``transform_input=False`` so
       the KM wrapper sees the same input the user provides; downstream
       pipelines are responsible for ImageNet normalization.

    5. **Channel-naming alignment with the OpenAI Microscope catalogue.**
       Torchvision's ``inception3a / 3b / 4a / 4b / 4c / 4d / 4e / 5a / 5b``
       correspond to Microscope's ``mixed3a / 3b / 4a / 4b / 4c / 4d / 4e
       / 5a / 5b`` and the Distill *Circuits Thread* (Olah, Cammarata et
       al., 2020) channel indices, which are preserved from the original
       Caffe export. Output channel counts per module:

           inception3a:  64 +  128 +  32 +  32 = 256
           inception3b: 128 +  192 +  96 +  64 = 480
           inception4a: 192 +  208 +  48 +  64 = 512
           inception4b: 160 +  224 +  64 +  64 = 512
           inception4c: 128 +  256 +  64 +  64 = 512
           inception4d: 112 +  288 +  64 +  64 = 528
           inception4e: 256 +  320 + 128 + 128 = 832
           inception5a: 256 +  320 + 128 + 128 = 832
           inception5b: 384 +  384 + 128 + 128 = 1024

    Linearizing parallel branches: each Inception module forks the module
    input across 4 branches that are later concatenated. We linearize via
    ``NN.branch_input(start, end)`` -- after each branch finishes, the
    next branch starts by restoring x to the snapshot of the module's
    fork point. The first 3 branch outputs are captured at branch
    boundaries (which double as ``concat_skip`` sources for the next
    module's first layer); the 4th branch's output naturally is the
    "current x" when the next module starts. ``concat_skip`` then
    assembles the 4-way merge at the next module's first layer.
"""
from typing import Tuple, Union

from torch import nn
from torchvision.models import googlenet, GoogLeNet_Weights
from torchvision.models.googlenet import (
    GoogLeNet as _TVGoogLeNet,
    BasicConv2d as _TVBasicConv2d,
    Inception as _TVInception,
)

from knowledgematrix.neural_net import NN


class GoogLeNet(NN):
    """
        GoogLeNet (Inception v1) model with linearized parallel branches.

        Args:
            input_shape (Tuple[int]): Shape of the input to the network
                (C, H, W). For pretrained variants, the canonical shape is
                (3, 224, 224); the architecture's strided convs / pools
                require H, W >= 224 to avoid degenerate spatial maps.
            num_classes (int): Number of output classes.
            save (bool): Whether to save activations and preactivations
                (required for matrix-computation forward).
            pretrained (bool): If True, load torchvision googlenet
                IMAGENET1K_V1 weights.
            pretrained_model (Union[_TVGoogLeNet, None]): Optional
                torchvision GoogLeNet instance to use as the source of
                weights. If None and ``pretrained`` is True, the default
                IMAGENET1K_V1 with ``aux_logits=False`` is used.
            device (str): The device to run the network on.
    """
    def __init__(
            self,
            input_shape: Tuple[int],
            num_classes: int = 1000,
            save: bool = False,
            pretrained: bool = False,
            pretrained_model: Union[_TVGoogLeNet, None] = None,
            device: str = "cpu",
        ) -> None:
        super().__init__(input_shape, save, device)
        # Sources captured at branch boundaries of the most recently built
        # inception module. Wired into the next module's first layer (or
        # the final pooling layer) via concat_skip to produce the 4-way
        # channel concatenation that is the module's output.
        self._pending_sources: list[int] = []

        if pretrained:
            if pretrained_model is None:
                # transform_input=False so KM input == torchvision input.
                # NOTE: torchvision's googlenet() builder forces
                # aux_logits=True when loading the pretrained weights
                # (the checkpoint contains aux head weights), then sets
                # aux1/aux2 to None automatically when the user-supplied
                # aux_logits was False. Passing aux_logits=False directly
                # raises ValueError. So we omit it and torchvision strips
                # the aux heads itself.
                pretrained_model = googlenet(
                    weights=GoogLeNet_Weights.IMAGENET1K_V1,
                    transform_input=False,
                )
                # Defensive: ensure aux heads truly off (torchvision
                # already does this when the original aux_logits arg was
                # missing/False, but make it explicit).
                pretrained_model.aux_logits = False
                pretrained_model.aux1 = None
                pretrained_model.aux2 = None
            else:
                if not isinstance(pretrained_model, _TVGoogLeNet):
                    raise ValueError("pretrained_model must be an instance of torchvision GoogLeNet.")
                # Make sure aux heads are off and transform_input is off so
                # the eval-mode forward matches the wrapper.
                pretrained_model.aux_logits = False
                pretrained_model.aux1 = None
                pretrained_model.aux2 = None
                pretrained_model.transform_input = False
            pretrained_model.eval()
            self._build_from_pretrained(pretrained_model)
            stem_out = self.layers[0].out_channels
            classifier_in = self.layers[-1].in_features
            if input_shape[0] != 3:
                print(f"Warning: The pretrained model was trained on 3-channel images. The input shape is {input_shape}. The first layer won't have pretrained weights.")
                self.layers[0] = nn.Conv2d(input_shape[0], stem_out, kernel_size=7, stride=2, padding=3, bias=False)
            elif num_classes != 1000:
                print(f"Warning: The pretrained model was trained on 1000 classes. The number of classes is {num_classes}. The last layer won't have pretrained weights.")
                self.layers[-1] = nn.Linear(classifier_in, num_classes)
        else:
            self._build_custom(input_shape[0], num_classes)


    ### Helper methods ###

    def _basic_conv(
            self,
            in_ch: int,
            out_ch: int,
            kernel_size,
            stride=(1, 1),
            padding=(0, 0),
        ) -> None:
        # BasicConv2d in torchvision: Conv2d (no bias) -> BatchNorm2d
        # (eps=0.001) -> ReLU. Match the BN eps to torchvision's default
        # for googlenet.
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
        # Append a torchvision BasicConv2d (Conv2d + BatchNorm2d) plus a
        # fresh ReLU. BasicConv2d.forward applies F.relu, equivalent to
        # nn.ReLU().
        self.layers.append(basic.conv)
        self.layers.append(basic.bn)
        self.relu()

    def _maxpool_ceil(
            self,
            kernel_size: int,
            stride: int,
            padding: int = 0,
        ) -> None:
        """
            Append a MaxPool2d with ``ceil_mode=True`` (which the NN base
            class's ``self.maxpool()`` does not support). Required for
            torchvision's GoogLeNet stem and inter-stage pools.
        """
        self.layers.append(nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=True,
            return_indices=True,
        ))

    def _wire_pending_merge(self, target: int) -> None:
        """Wire the pending branch outputs into ``target`` via
        ``concat_skip``, producing the prior module's 4-way merge as the
        input at ``target``."""
        for s in self._pending_sources:
            self.concat_skip(s, target)
        self._pending_sources = []


    ### Custom (parameterized) GoogLeNet builder ###

    def _build_custom(self, in_channels: int, num_classes: int) -> None:
        # Stem: conv1 (7x7 s2) -> maxpool (3 s2 ceil) -> conv2 (1x1) ->
        #       conv3 (3x3) -> maxpool (3 s2 ceil)
        self._basic_conv(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self._maxpool_ceil(kernel_size=3, stride=2)
        self._basic_conv(64, 64, kernel_size=1)
        self._basic_conv(64, 192, kernel_size=3, padding=1)
        self._maxpool_ceil(kernel_size=3, stride=2)

        # 9 inception modules with 2 inter-stage maxpools.
        # (in_ch, b1, b2_red, b2, b3_red, b3, b_pool) per torchvision spec.
        in_ch = 192
        in_ch = self._build_inception(in_ch,  64,  96, 128, 16, 32, 32)   # 3a -> 256
        in_ch = self._build_inception(in_ch, 128, 128, 192, 32, 96, 64)   # 3b -> 480
        # maxpool3
        target = self.get_num_layers()
        self._wire_pending_merge(target)
        self._maxpool_ceil(kernel_size=3, stride=2)

        in_ch = self._build_inception(in_ch, 192,  96, 208, 16, 48, 64)   # 4a -> 512
        in_ch = self._build_inception(in_ch, 160, 112, 224, 24, 64, 64)   # 4b -> 512
        in_ch = self._build_inception(in_ch, 128, 128, 256, 24, 64, 64)   # 4c -> 512
        in_ch = self._build_inception(in_ch, 112, 144, 288, 32, 64, 64)   # 4d -> 528
        in_ch = self._build_inception(in_ch, 256, 160, 320, 32, 128, 128) # 4e -> 832
        # maxpool4 (note: torchvision uses kernel_size=2 here, NOT 3)
        target = self.get_num_layers()
        self._wire_pending_merge(target)
        self._maxpool_ceil(kernel_size=2, stride=2)

        in_ch = self._build_inception(in_ch, 256, 160, 320, 32, 128, 128) # 5a -> 832
        in_ch = self._build_inception(in_ch, 384, 192, 384, 48, 128, 128) # 5b -> 1024

        # Final head: AdaptiveAvgPool -> Flatten -> Linear (concat_skip
        # merges the last inception module's branches into AdaptiveAvgPool).
        # Dropout is identity in eval mode and is omitted (matches
        # inception_v3 wrapper convention).
        target = self.get_num_layers()
        self._wire_pending_merge(target)
        self.adaptiveavgpool((1, 1))
        self.flatten()
        self.linear(in_ch, num_classes)


    ### Inception module builders (custom) ###

    def _build_inception(
            self,
            in_ch: int,
            b1: int,
            b2_red: int,
            b2: int,
            b3_red: int,
            b3: int,
            b_pool: int,
        ) -> int:
        """
            Build one Inception V1 module: 4 parallel branches concatenated.
            Returns the output channel count = b1 + b2 + b3 + b_pool.

            Branches (matching torchvision.models.googlenet.Inception):
              branch1: 1x1 conv (b1 channels)
              branch2: 1x1 conv (b2_red) -> 3x3 conv padding=1 (b2)
              branch3: 1x1 conv (b3_red) -> 3x3 conv padding=1 (b3)  [the
                       known-bug 3x3 instead of 5x5; we match torchvision]
              branch4: 3x3 maxpool stride=1 padding=1 ceil_mode=True
                       -> 1x1 conv (b_pool)
        """
        fork = self.get_num_layers()
        self._wire_pending_merge(fork)

        # Branch 1: 1x1 conv
        self._basic_conv(in_ch, b1, kernel_size=1)

        # Branch 2: 1x1 reduce -> 3x3 conv
        b2_start = self.get_num_layers()
        self.branch_input(fork, b2_start)
        self._basic_conv(in_ch, b2_red, kernel_size=1)
        self._basic_conv(b2_red, b2, kernel_size=3, padding=1)

        # Branch 3: 1x1 reduce -> 3x3 conv (torchvision quirk: 3x3, not 5x5)
        b3_start = self.get_num_layers()
        self.branch_input(fork, b3_start)
        self._basic_conv(in_ch, b3_red, kernel_size=1)
        self._basic_conv(b3_red, b3, kernel_size=3, padding=1)

        # Branch 4: maxpool (3x3, stride=1, padding=1, ceil_mode=True) -> 1x1 conv
        b4_start = self.get_num_layers()
        self.branch_input(fork, b4_start)
        self._maxpool_ceil(kernel_size=3, stride=1, padding=1)
        self._basic_conv(in_ch, b_pool, kernel_size=1)

        # Sources for the next module's concat (in cat-order):
        #   b2_start -> branch1 output
        #   b3_start -> branch2 output
        #   b4_start -> branch3 output
        # Plus current x at next module start = branch4 output.
        self._pending_sources = [b2_start, b3_start, b4_start]
        return b1 + b2 + b3 + b_pool


    ### Pretrained (torchvision) GoogLeNet builder ###

    def _build_from_pretrained(self, pretrained_model: _TVGoogLeNet) -> None:
        # Stem
        self._append_basic_conv(pretrained_model.conv1)
        self._maxpool_ceil(
            kernel_size=pretrained_model.maxpool1.kernel_size,
            stride=pretrained_model.maxpool1.stride,
            padding=pretrained_model.maxpool1.padding,
        )
        self._append_basic_conv(pretrained_model.conv2)
        self._append_basic_conv(pretrained_model.conv3)
        self._maxpool_ceil(
            kernel_size=pretrained_model.maxpool2.kernel_size,
            stride=pretrained_model.maxpool2.stride,
            padding=pretrained_model.maxpool2.padding,
        )

        # Inception 3a, 3b
        self._append_inception(pretrained_model.inception3a)
        self._append_inception(pretrained_model.inception3b)
        # maxpool3
        target = self.get_num_layers()
        self._wire_pending_merge(target)
        self._maxpool_ceil(
            kernel_size=pretrained_model.maxpool3.kernel_size,
            stride=pretrained_model.maxpool3.stride,
            padding=pretrained_model.maxpool3.padding,
        )

        # Inception 4a, 4b, 4c, 4d, 4e
        self._append_inception(pretrained_model.inception4a)
        self._append_inception(pretrained_model.inception4b)
        self._append_inception(pretrained_model.inception4c)
        self._append_inception(pretrained_model.inception4d)
        self._append_inception(pretrained_model.inception4e)
        # maxpool4
        target = self.get_num_layers()
        self._wire_pending_merge(target)
        self._maxpool_ceil(
            kernel_size=pretrained_model.maxpool4.kernel_size,
            stride=pretrained_model.maxpool4.stride,
            padding=pretrained_model.maxpool4.padding,
        )

        # Inception 5a, 5b
        self._append_inception(pretrained_model.inception5a)
        self._append_inception(pretrained_model.inception5b)

        # Final head: AdaptiveAvgPool -> Flatten -> fc.
        # Dropout (p=0.2) is identity in eval mode and is omitted.
        target = self.get_num_layers()
        self._wire_pending_merge(target)
        self.adaptiveavgpool(pretrained_model.avgpool.output_size)
        self.flatten()
        self.layers.append(pretrained_model.fc)


    ### Inception module appender (pretrained) ###

    def _append_inception(self, m: _TVInception) -> None:
        """
            Append a torchvision Inception module by re-using its
            ``branch1``/``branch2``/``branch3``/``branch4`` sub-modules
            so the pretrained weights are preserved. Layout matches
            ``torchvision.models.googlenet.Inception``:

              branch1: BasicConv2d (1x1)
              branch2: Sequential(BasicConv2d 1x1, BasicConv2d 3x3 pad=1)
              branch3: Sequential(BasicConv2d 1x1, BasicConv2d 3x3 pad=1)
              branch4: Sequential(MaxPool2d 3 s1 p1 ceil, BasicConv2d 1x1)
        """
        fork = self.get_num_layers()
        self._wire_pending_merge(fork)

        # Branch 1: single 1x1 BasicConv2d
        self._append_basic_conv(m.branch1)

        # Branch 2: 1x1 -> 3x3
        b2_start = self.get_num_layers()
        self.branch_input(fork, b2_start)
        for sub in m.branch2:
            self._append_basic_conv(sub)

        # Branch 3: 1x1 -> 3x3
        b3_start = self.get_num_layers()
        self.branch_input(fork, b3_start)
        for sub in m.branch3:
            self._append_basic_conv(sub)

        # Branch 4: maxpool (ceil_mode=True) -> 1x1 BasicConv2d
        b4_start = self.get_num_layers()
        self.branch_input(fork, b4_start)
        # m.branch4[0] is nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True)
        pool = m.branch4[0]
        self._maxpool_ceil(
            kernel_size=pool.kernel_size,
            stride=pool.stride,
            padding=pool.padding,
        )
        # m.branch4[1] is the BasicConv2d(in, pool_proj, 1x1)
        self._append_basic_conv(m.branch4[1])

        self._pending_sources = [b2_start, b3_start, b4_start]


if __name__ == "__main__":
    import torch
    print("=== Forward smoke ===")
    m = GoogLeNet(input_shape=(3, 224, 224), num_classes=1000, pretrained=True, device="cpu")
    m.eval()
    out = m(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 1000), out.shape
    print("forward OK", out.shape)

    print("=== KM shape smoke ===")
    from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
    km = KnowledgeMatrixComputer(m, batch_size=128, device="cpu").forward(torch.randn(3, 224, 224))
    assert km.shape == (1000, 150529), km.shape
    print("KM shape OK", km.shape)

    print("=== Completeness check (M(x).sum(1) ?= f(x)) ===")
    x = torch.randn(3, 224, 224)
    logit2 = m(x.unsqueeze(0)).squeeze(0)
    km2 = KnowledgeMatrixComputer(m, batch_size=128, device="cpu").forward(x)
    err = (km2.sum(1) - logit2).abs().max().item()
    print("max |rowsum - logit|:", err)
    # Expect ~1e-5 (fp32). If much larger, the wrapper is wrong.

from __future__ import annotations

import copy
import operator
from collections import deque

import torch
from torch import nn
from torch.nn import functional as F
import torch.fx as fx
import math
from typing import Union, Dict, Tuple, List, Optional


class NN(nn.Module):
    """
        A class to build a neural network for which the knowledge matrix can be computed.

        Args:
            input_shape (Tuple[int]): The shape of the input to the network.
            save (bool): Whether to save the activations and preactivations of the network.
            device (str): The device to run the network on.
    """

    def __init__(
            self, 
            input_shape: Tuple[int],
            save: bool=False, 
            device: str="cpu"
        ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.save = save
        self.device = device
        self.layers = nn.ModuleList()
        self.residuals: Dict[int, Tuple[int, list[nn.Module]]] = {}
        self.residuals_starts: set[int] = set()
        self.residual_modules = nn.ModuleList()
        # Concatenation skip connections (used by DenseNet, U-Net):
        #   concat_skips[end] -> ordered list of source layer indices whose
        #   captured tensors are concatenated (along channel dim) BEFORE x at
        #   layer `end`. concat_skips_starts holds every source index so
        #   forward() snapshots x at the right moment.
        self.concat_skips: Dict[int, list[int]] = {}
        self.concat_skips_starts: set[int] = set()
        # Branch-input wiring (used by Inception): at layer `end`, REPLACE x
        # with the snapshot captured at layer `start` (i.e. discard whatever
        # value flowed through the previous branch). This linearizes the
        # parallel branches of an Inception module: branch 1 runs naturally,
        # branch_input restores x to the module's fork-point at the start of
        # branch 2, 3, ..., and the eventual concat at the merge layer is
        # handled by concat_skip.
        self.branch_inputs: Dict[int, int] = {}
        self.branch_inputs_starts: set[int] = set()


    ### Linear Layers ###

    def linear(
            self, 
            in_features: int, 
            out_features: int, 
            bias: bool=True
        ) -> None:
        self.layers.append(nn.Linear(
            in_features=in_features, 
            out_features=out_features, 
            bias=bias
        ))
    
    def conv(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int],
            stride: Tuple[int]=(1,1),
            padding: Tuple[int]=(0,0),
            dilation: Tuple[int]=(1,1),
            groups: int=1,
            bias: bool=True
        ) -> None:
        self.layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        ))
    
    def conv1d(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int=1,
            padding: int=0,
            dilation: int=1,
            groups: int=1,
            bias: bool=True
        ) -> None:
        self.layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size,1),
            stride=(stride,1),
            padding=(padding,0),
            dilation=(dilation,1),
            groups=groups,
            bias=bias
        ))

    def conv_transpose(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int],
            stride: Tuple[int]=(1,1),
            padding: Tuple[int]=(0,0),
            output_padding: Tuple[int]=(0,0),
            dilation: Tuple[int]=(1,1),
            groups: int=1,
            bias: bool=True
        ) -> None:
        self.layers.append(nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        ))

    def conv_transpose1d(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int=1,
            padding: int=0,
            output_padding: int=0,
            dilation: int=1,
            groups: int=1,
            bias: bool=True
        ) -> None:
        self.layers.append(nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size,1),
            stride=(stride,1),
            padding=(padding,0),
            output_padding=(output_padding,0),
            dilation=(dilation,1),
            groups=groups,
            bias=bias
        ))
    
    def flatten(
            self,
            start_dim: int=1,
            end_dim: int=-1) -> None:
        self.layers.append(nn.Flatten(
            start_dim=start_dim,
            end_dim=end_dim
        ))
    
    def embedding(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: Union[int,None]=None,
            max_norm: Union[float,None]=None,
            norm_type: float=2.0,
            scale_grad_by_freq: bool=False,
            sparse: bool=False
    ) -> None:
        self.layers.append(nn.Embedding(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse
        ))

    
    ### Normalization Layers ###

    def batchnorm(
            self,
            num_features: int,
            eps: float=0.00001,
            momentum: float=0.1
    ) -> None:
        self.layers.append(nn.BatchNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum
        ))
    
    def batchnorm1d(
            self,
            num_features: int,
            eps: float=0.00001,
            momentum: float=0.1
    ) -> None:
        self.batchnorm(num_features, eps, momentum)
    
    def layernorm(
            self,
            normalized_shape: Union[int,Tuple[int],torch.Size],
            eps: float=1e-5,
            elementwise_affine: bool=True,
            bias: bool=True
    ) -> None:
        self.layers.append(nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias
        ))

    def rmsnorm(self, normalized_shape: int, eps: float = 1e-6) -> None:
        self.layers.append(RMSNorm(normalized_shape=normalized_shape, eps=eps))

    def groupnorm(
            self,
            num_groups: int,
            num_channels: int,
            eps: float = 1e-5,
            affine: bool = True
    ) -> None:
        self.layers.append(nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine
        ))

    def instancenorm(
            self,
            num_channels: int,
            eps: float = 1e-5,
            affine: bool = True
    ) -> None:
        self.layers.append(nn.GroupNorm(
            num_groups=num_channels,
            num_channels=num_channels,
            eps=eps,
            affine=affine
        ))


    ### Pooling Layers ###

    def avgpool(
            self,
            kernel_size: Tuple[int],
            stride: Union[Tuple[int],None]=None,
            padding: Tuple[int]=(0,0)
    ) -> None:
        if stride is None:
            stride = kernel_size
        self.layers.append(nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ))
    
    def avgpool1d(
            self,
            kernel_size: int,
            stride: Union[int,None]=None,
            padding: int=0
    ) -> None:
        if stride is None:
            stride = kernel_size
        self.layers.append(nn.AvgPool2d(
            kernel_size=(kernel_size,1),
            stride=(stride,1),
            padding=(padding,0)
        ))
    
    def adaptiveavgpool(
            self,
            output_size: int
    ) -> None:
        self.layers.append(nn.AdaptiveAvgPool2d(output_size=output_size))
    
    def adaptiveavgpool1d(
            self,
            output_size: int
    ) -> None:
        self.layers.append(nn.AdaptiveAvgPool2d(output_size=(output_size,1)))
    
    def maxpool(
            self, 
            kernel_size: Tuple[int],
            stride: Union[Tuple[int],None]=None,
            padding: Tuple[int]=(0,0)
        ) -> None:
        if stride is None:
            stride = kernel_size
        self.layers.append(nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_indices=True
        ))
    
    def maxpool1d(
            self, 
            kernel_size: int,
            stride: Union[int,None]=None,
            padding: int=0
        ) -> None:
        if stride is None:
            stride = kernel_size
        self.layers.append(nn.MaxPool2d(
            kernel_size=(kernel_size,1),
            stride=(stride,1),
            padding=(padding,0),
            return_indices=True
        ))
    
    def adaptivemaxpool(
            self,
            output_size: int
    ) -> None:
        self.layers.append(nn.AdaptiveMaxPool2d(output_size=output_size, return_indices=True))

    def adaptivemaxpool1d(
            self,
            output_size: int
    ) -> None:
        self.layers.append(nn.AdaptiveMaxPool2d(output_size=(output_size,1), return_indices=True))


    ### Upsampling Layers ###

    def upsample(
            self,
            scale_factor: Union[int, float, Tuple[int]],
            mode: str = 'nearest'
    ) -> None:
        kwargs = {'scale_factor': scale_factor, 'mode': mode}
        if mode in ('bilinear', 'bicubic', 'trilinear'):
            kwargs['align_corners'] = False
        self.layers.append(nn.Upsample(**kwargs))

    def pixel_shuffle(self, upscale_factor: int) -> None:
        self.layers.append(nn.PixelShuffle(upscale_factor=upscale_factor))


    ### Dropout ###

    def dropout(self, p: float=0.5) -> None:
        self.layers.append(nn.Dropout(p=p))


    ### Activation Functions ###

    def elu(self, alpha: float=1) -> None:
        self.layers.append(nn.ELU(alpha=alpha))

    def gelu(self, approximate: str="none") -> None:
        self.layers.append(nn.GELU(approximate=approximate))

    def leakyrelu(self, negative_slope: float=0.01) -> None:
        self.layers.append(nn.LeakyReLU(negative_slope=negative_slope))

    def relu(self) -> None:
        self.layers.append(nn.ReLU())
    
    def sigmoid(self) -> None:
        self.layers.append(nn.Sigmoid())

    def silu(self) -> None:
        self.layers.append(nn.SiLU())

    def mish(self) -> None:
        self.layers.append(nn.Mish())

    def softmax(self, dim: Union[int,None]=None) -> None:
        self.layers.append(nn.Softmax(dim=dim))

    def tanh(self) -> None:
        self.layers.append(nn.Tanh())

    def celu(self, alpha: float = 1.0) -> None:
        self.layers.append(nn.CELU(alpha=alpha))

    def hardsigmoid(self) -> None:
        self.layers.append(nn.Hardsigmoid())

    def hardswish(self) -> None:
        self.layers.append(nn.Hardswish())

    def prelu(self, num_parameters: int = 1, init: float = 0.25) -> None:
        self.layers.append(nn.PReLU(num_parameters=num_parameters, init=init))

    def relu6(self) -> None:
        self.layers.append(nn.ReLU6())

    def softplus(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        self.layers.append(nn.Softplus(beta=beta, threshold=threshold))

    def jumprelu(self, threshold: torch.Tensor) -> None:
        self.layers.append(JumpReLU(threshold=threshold))

    def topk_activation(self, k: int) -> None:
        self.layers.append(TopKActivation(k=k))

    def multiheadattention(
            self,
            d_model: int,
            num_heads: int,
            num_kv_heads: Union[int,None]=None,
            mask: Union[torch.Tensor,None]=None
        ) -> None:
        self.layers.append(
            MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mask=mask
            )
        )


    ### Positional Encoding ###

    def positionalencoding(self, d_model: int, max_len: int=5000) -> None:
        self.layers.append(PositionalEncoding(d_model, max_len))


    ### Residual Connections ###

    def residual(self, start: int, end: int) -> None:
        shape_start = self.shape_at_layer(start)
        shape_end = self.shape_at_layer(end)
        if (start < end):
            if shape_start == shape_end:
                projection = [nn.Identity()]
            else:
                if len(shape_start) == len(shape_end):
                    if len(shape_start) >= 4 and len(shape_end) >= 4:  # Conv
                        projection = [
                            nn.Conv2d(
                                shape_start[1], 
                                shape_end[1], 
                                kernel_size=1,
                                stride = (
                                    round(shape_start[2] / shape_end[2]),
                                    round(shape_start[3] / shape_end[3]),
                                ),
                                bias=False
                            ),
                            nn.BatchNorm2d(shape_end[1])
                        ]
                    elif len(shape_start) <= 3 and len(shape_end) <= 3:  # FC
                        projection = [
                            nn.Linear(
                                shape_start[-1],
                                shape_end[-1],
                                bias=True
                            )
                        ]
                else:
                    raise ValueError(f"The lenghts of shape at layer {start} and {end} need to be equal to have a residual connection. Got {shape_start} and {shape_end}.")
        else:
            raise ValueError(f"To have a residual connection from layer {start} to {end}, one needs {start} < {end}.")
        for module in projection:
            if not isinstance(module, nn.Identity):
                self.residual_modules.append(module)
        self.residuals_starts.add(start)
        if end in self.residuals:
                self.residuals[end].append((start, projection))
        else:
            self.residuals[end] = [(start, projection)]

    def concat_skip(self, start: int, end: int) -> None:
        """
            Register a channel-wise concatenation skip: the tensor captured
            at `start` is concatenated with the tensor arriving at `end`
            along dim=1 (channels), in call order with the current x last.
            Sources must share spatial dims with x at `end`. No projection
            is applied -- the user is responsible for sizing downstream
            layers to receive the grown channel dim.
        """
        if start >= end:
            raise ValueError(f"concat_skip requires start < end, got start={start}, end={end}.")
        self.concat_skips_starts.add(start)
        if end in self.concat_skips:
            self.concat_skips[end].append(start)
        else:
            self.concat_skips[end] = [start]

    def apply_concat(self, x: torch.Tensor, outputs: list[torch.Tensor], layer: int) -> torch.Tensor:
        sources = [outputs[s] for s in self.concat_skips[layer]]
        return torch.cat(sources + [x], dim=1)

    def branch_input(self, start: int, end: int) -> None:
        """
            Register a branch-input wiring: at layer `end`, REPLACE x with
            the tensor captured at layer `start`. Used to linearize the
            parallel branches of an Inception module -- after branch k
            finishes, branch k+1 starts from the fork-point snapshot rather
            than from branch k's output. The current x at `end` is dropped
            (its value is typically captured separately as branch k's output
            via concat_skip's start-snapshot mechanism, since `end` will
            usually be in concat_skips_starts too).
        """
        if start >= end:
            raise ValueError(f"branch_input requires start < end, got start={start}, end={end}.")
        if end in self.branch_inputs:
            raise ValueError(f"branch_input destination {end} already set to {self.branch_inputs[end]}.")
        self.branch_inputs_starts.add(start)
        self.branch_inputs[end] = start


    ### Conversion from an arbitrary torch Module (MVP) ###

    @classmethod
    def from_torch(
            cls,
            module: nn.Module,
            input_shape: Tuple[int, ...],
            num_classes: Union[int, None] = None,
            device: str = "cpu",
        ) -> "NN":
        """
            Convert an in-scope pretrained torchvision ``nn.Module`` into an
            ``NN`` whose ``forward`` exactly reproduces the source (so its
            knowledge matrix can be computed).

            MVP scope: pure-sequential nets (VGG, AlexNet) and post-activation
            residual nets that merge via ``operator.add`` with a post-add
            activation (ResNet-18/34/50/101/152). Anything else -- concat
            (DenseNet), branch (Inception), SE/``mul`` gating, attention,
            reshape/permute, linear-bottleneck/pre-activation residuals -- is
            out of scope and raises ``NotImplementedError`` rather than
            silently mis-wiring.

            Pipeline: ``torch.fx.symbolic_trace`` -> classify each node
            (LAYER / WIRING / DROP / UNMAPPABLE) -> emit into a fresh ``NN``,
            resolving residual skip endpoints to integer layer indices via a
            ``live_index`` map (first emitted-layer consumer of a tensor).

            Args:
                module: the source ``nn.Module`` (left untouched -- all
                    parametric submodules are deep-copied).
                input_shape: ``(C, H, W)`` shape of a single input sample.
                num_classes: if given and the final ``Linear`` has a different
                    ``out_features``, the head is replaced with a fresh
                    ``Linear`` of this width (no pretrained weights).
                device: device for the resulting ``NN``.
        """
        gm = fx.symbolic_trace(module)
        nodes = list(gm.graph.nodes)

        # ---- Pass 0: find residual-merge nodes and their skip/downsample paths.
        merge_nodes: List[fx.Node] = [
            n for n in nodes
            if _ft_is_add(n) and len(_ft_tensor_inputs(n)) == 2
        ]
        merge_set = set(merge_nodes)
        merge_info: Dict[fx.Node, Tuple[fx.Node, List[nn.Module]]] = {}
        skip_node_set: set = set()
        for m in merge_nodes:
            a, b = _ft_tensor_inputs(m)
            fork = _ft_fork(a, b)
            if fork is None:
                raise NotImplementedError(
                    f"from_torch: cannot resolve a common fork for residual add "
                    f"at node {m.name}."
                )
            da, db = _ft_dist(a, fork), _ft_dist(b, fork)
            # main = the deeper branch; skip = the shallower (identity/downsample).
            skip_arg = b if da >= db else a
            skip_path = _ft_linear_path(skip_arg, fork, m)
            skip_src: List[nn.Module] = []
            for sn in skip_path:
                if sn.op != "call_module":
                    raise NotImplementedError(
                        f"from_torch: residual skip path at node {m.name} contains a "
                        f"non-module op {sn.op} ({sn.target}); out of MVP scope."
                    )
                sub = gm.get_submodule(sn.target)
                if not isinstance(sub, (nn.Conv2d, nn.BatchNorm2d)):
                    raise NotImplementedError(
                        f"from_torch: unsupported downsample/projection module "
                        f"{type(sub).__name__} at node {sn.name}; out of MVP scope."
                    )
                skip_src.append(sub)
            merge_info[m] = (fork, skip_src)
            skip_node_set.update(skip_path)

        # ---- Pass 1: emit layers; build emit_index / live_index / passthrough.
        self = cls(input_shape, save=False, device=device)
        live_index: Dict[fx.Node, int] = {}
        passthrough: Dict[fx.Node, fx.Node] = {}

        def resolve(n: fx.Node) -> fx.Node:
            seen = set()
            while n in passthrough and n not in seen:
                seen.add(n)
                n = passthrough[n]
            return n

        for node in nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node in skip_node_set or node in merge_set:
                # Skip/downsample modules and residual adds are handled as
                # wiring in pass 2, not emitted on the main line.
                continue
            status = self._ft_emit(node, gm)
            if status == "drop":
                ins = _ft_tensor_inputs(node)
                passthrough[node] = resolve(ins[0]) if ins else node
                continue
            # A layer was appended; record its index and its consumption of
            # upstream tensors (first emitted consumer wins -> live_index).
            idx = self.get_num_layers() - 1
            for inp in _ft_tensor_inputs(node):
                src = resolve(inp)
                if src not in live_index:
                    live_index[src] = idx

        # ---- Pass 2: register residual wiring; guard the two hazards.
        for m in merge_nodes:
            fork, skip_src = merge_info[m]
            fork_r = resolve(fork)
            # Hazard 1: terminal add (no post-merge layer consumes it).
            if m not in live_index:
                raise NotImplementedError(
                    f"from_torch: terminal residual add at node {m.name} (no "
                    f"following layer); NN.forward would silently drop it."
                )
            end = live_index[m]
            # Hazard 2: no-gap adjacent merge. The fork must resolve to an
            # emitted-layer output (or the network input), never a raw merge
            # output -- otherwise the fork snapshot captures the pre-add tensor.
            if fork_r in merge_set:
                raise NotImplementedError(
                    f"from_torch: residual at node {m.name} forks from an "
                    f"un-activated merge output (no post-merge activation between "
                    f"adjacent residual blocks). Architectures such as "
                    f"MobileNetV2/EfficientNet linear bottlenecks and "
                    f"pre-activation ResNets are out of MVP scope."
                )
            if fork_r not in live_index:
                raise NotImplementedError(
                    f"from_torch: cannot resolve residual fork index for node "
                    f"{m.name}."
                )
            start = live_index[fork_r]
            if not (start < end) or end >= self.get_num_layers():
                raise NotImplementedError(
                    f"from_torch: invalid residual endpoints (start={start}, "
                    f"end={end}) at node {m.name}."
                )
            self.residual(start, end)
            # Keep auto-projection BNs in eval so later shape_at_layer calls
            # don't corrupt their running stats.
            for _, proj in self.residuals[end]:
                for sub in proj:
                    if isinstance(sub, nn.BatchNorm2d):
                        sub.eval()
            # Override the auto projection with the real downsample modules.
            if skip_src:
                new_proj = [copy.deepcopy(s) for s in skip_src]
                for mod in new_proj:
                    if isinstance(mod, nn.BatchNorm2d):
                        mod.eval()
                    self.residual_modules.append(mod)
                overridden = False
                lst = self.residuals[end]
                for k, (s, _proj) in enumerate(lst):
                    if s == start:
                        lst[k] = (start, new_proj)
                        overridden = True
                        break
                assert overridden, (
                    f"from_torch: downsample projection override failed for node "
                    f"{m.name} (start={start}, end={end})."
                )

        # ---- Optional classifier-head swap.
        if num_classes is not None:
            last = self.layers[-1]
            if isinstance(last, nn.Linear) and last.out_features != num_classes:
                self.layers[-1] = nn.Linear(last.in_features, num_classes)

        self.to(device)
        return self

    def _ft_emit(self, node: fx.Node, gm: fx.GraphModule) -> str:
        """
            Emit one fx node into ``self.layers``. Returns "layer" if a layer
            was appended, "drop" if the node is a no-op at eval (Dropout /
            Identity). Raises ``NotImplementedError`` for anything unmappable.
        """
        if node.op == "call_module":
            return self._ft_emit_module(gm.get_submodule(node.target), node)
        if node.op in ("call_function", "call_method"):
            return self._ft_emit_func(node)
        raise NotImplementedError(
            f"from_torch: unsupported node op {node.op} at node {node.name}."
        )

    def _ft_emit_module(self, sub: nn.Module, node: fx.Node) -> str:
        # Drop no-ops (identity at eval).
        if isinstance(sub, (nn.Dropout, nn.Dropout1d, nn.Dropout2d,
                            nn.Dropout3d, nn.AlphaDropout, nn.Identity)):
            return "drop"
        # Parametric: deep-copy so weights/bias/BN stats transfer exactly and
        # the caller's module is never mutated.
        if isinstance(sub, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear,
                            nn.LayerNorm, nn.GroupNorm)):
            self.layers.append(copy.deepcopy(sub))
            return "layer"
        if isinstance(sub, nn.BatchNorm2d):
            bn = copy.deepcopy(sub)
            bn.eval()  # protect running stats from shape_at_layer's random forwards
            self.layers.append(bn)
            return "layer"
        if isinstance(sub, nn.PReLU):  # parametric activation
            self.layers.append(copy.deepcopy(sub))
            return "layer"
        # Stateless ops rebuilt via the NN builders.
        if isinstance(sub, nn.ReLU):
            self.relu(); return "layer"
        if isinstance(sub, nn.ReLU6):
            self.relu6(); return "layer"
        if isinstance(sub, nn.LeakyReLU):
            self.leakyrelu(sub.negative_slope); return "layer"
        if isinstance(sub, nn.ELU):
            self.elu(sub.alpha); return "layer"
        if isinstance(sub, nn.CELU):
            self.celu(sub.alpha); return "layer"
        if isinstance(sub, nn.SiLU):
            self.silu(); return "layer"
        if isinstance(sub, nn.GELU):
            self.gelu(sub.approximate); return "layer"
        if isinstance(sub, nn.Mish):
            self.mish(); return "layer"
        if isinstance(sub, nn.Sigmoid):
            self.sigmoid(); return "layer"
        if isinstance(sub, nn.Hardsigmoid):
            self.hardsigmoid(); return "layer"
        if isinstance(sub, nn.Hardswish):
            self.hardswish(); return "layer"
        if isinstance(sub, nn.Tanh):
            self.tanh(); return "layer"
        if isinstance(sub, nn.Softplus):
            self.softplus(sub.beta, sub.threshold); return "layer"
        if isinstance(sub, nn.Softmax):
            self.softmax(sub.dim); return "layer"
        if isinstance(sub, nn.MaxPool2d):
            self.maxpool(sub.kernel_size, sub.stride, sub.padding); return "layer"
        if isinstance(sub, nn.AdaptiveMaxPool2d):
            self.adaptivemaxpool(sub.output_size); return "layer"
        if isinstance(sub, nn.AvgPool2d):
            self.avgpool(sub.kernel_size, sub.stride, sub.padding); return "layer"
        if isinstance(sub, nn.AdaptiveAvgPool2d):
            self.adaptiveavgpool(sub.output_size); return "layer"
        if isinstance(sub, nn.Flatten):
            self.flatten(sub.start_dim, sub.end_dim); return "layer"
        raise NotImplementedError(
            f"from_torch: unsupported module {type(sub).__name__} at node "
            f"{node.name}."
        )

    def _ft_emit_func(self, node: fx.Node) -> str:
        tgt = node.target
        is_method = node.op == "call_method"
        # Flatten.
        if tgt in (torch.flatten,) or (is_method and tgt == "flatten"):
            start_dim = _ft_arg(node, 1, "start_dim", 1)
            end_dim = _ft_arg(node, 2, "end_dim", -1)
            self.flatten(start_dim, end_dim); return "layer"
        # Functional activations.
        if tgt in (F.relu, torch.relu) or (is_method and tgt in ("relu", "relu_")):
            self.relu(); return "layer"
        if tgt is F.relu6:
            self.relu6(); return "layer"
        if tgt is F.leaky_relu:
            self.leakyrelu(_ft_arg(node, 1, "negative_slope", 0.01)); return "layer"
        if tgt is F.elu:
            self.elu(_ft_arg(node, 1, "alpha", 1.0)); return "layer"
        if tgt is F.silu:
            self.silu(); return "layer"
        if tgt is F.gelu:
            self.gelu(_ft_arg(node, 1, "approximate", "none")); return "layer"
        if tgt is F.mish:
            self.mish(); return "layer"
        if tgt in (F.sigmoid, torch.sigmoid) or (is_method and tgt == "sigmoid"):
            self.sigmoid(); return "layer"
        if tgt is F.hardsigmoid:
            self.hardsigmoid(); return "layer"
        if tgt is F.hardswish:
            self.hardswish(); return "layer"
        if tgt in (F.tanh, torch.tanh) or (is_method and tgt == "tanh"):
            self.tanh(); return "layer"
        if tgt is F.softmax or (is_method and tgt == "softmax"):
            self.softmax(_ft_arg(node, 1, "dim", None)); return "layer"
        # Functional pooling.
        if tgt is F.max_pool2d:
            self.maxpool(_ft_arg(node, 1, "kernel_size"),
                         _ft_arg(node, 2, "stride", None),
                         _ft_arg(node, 3, "padding", 0)); return "layer"
        if tgt is F.avg_pool2d:
            self.avgpool(_ft_arg(node, 1, "kernel_size"),
                         _ft_arg(node, 2, "stride", None),
                         _ft_arg(node, 3, "padding", 0)); return "layer"
        if tgt is F.adaptive_avg_pool2d:
            self.adaptiveavgpool(_ft_arg(node, 1, "output_size")); return "layer"
        if tgt is F.adaptive_max_pool2d:
            self.adaptivemaxpool(_ft_arg(node, 1, "output_size")); return "layer"
        raise NotImplementedError(
            f"from_torch: unsupported op {tgt} at node {node.name}."
        )


    ### Forward Method ###

    def forward(self, x: torch.Tensor, return_penultimate:bool=False) -> torch.Tensor:
        start_layer = self._get_start_layer()  # Start layer is the one after the embedding and positional encoding
        for layer in self.layers[:start_layer]:
            x = layer(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        # Update the input shape, useful when the input shape is not known beforehand (e.g. for transformers)
        self.input_shape = (x.shape[1], x.shape[2], x.shape[3])
        inputs_residuals: list[torch.Tensor] = [None] * self.get_num_layers()
        # branch_snapshots holds POST-concat snapshots for branch_input sources;
        # inputs_residuals holds PRE-concat snapshots for residual / concat_skip sources.
        branch_snapshots: list[torch.Tensor] = [None] * self.get_num_layers()
        if not self.save:  # Regular forward pass
            layers = self.layers[:-1] if return_penultimate else self.layers
            for i, layer in enumerate(layers[start_layer:], start=start_layer):
                if i in self.residuals_starts or i in self.concat_skips_starts:
                    inputs_residuals[i] = x
                if i in self.branch_inputs:
                    x = branch_snapshots[self.branch_inputs[i]]
                if i in self.residuals:
                    x = self.apply_residual(x, inputs_residuals, layer=i)
                if i in self.concat_skips:
                    x = self.apply_concat(x, inputs_residuals, layer=i)
                if i in self.branch_inputs_starts:
                    branch_snapshots[i] = x
                if isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                    x, _ = layer(x)
                else:
                    x = layer(x)
        else:  # Forward pass for matrix computation
               # Save activations and preactivations
            if return_penultimate:
                raise ValueError("return_penultimate is not supported for matrix computation.")
            self.pre_acts: list[torch.Tensor] = [None] * self.get_num_layers()
            self.acts: list[torch.Tensor] = [None] * self.get_num_layers()
            self.maxpool_indices: list[torch.Tensor] = [None] * self.get_num_layers()
            self.layernorms: list[torch.Tensor] = [None] * self.get_num_layers()

            for i, layer in enumerate(self.layers[start_layer:], start=start_layer):
                if i in self.residuals_starts or i in self.concat_skips_starts:
                    inputs_residuals[i] = x
                if i in self.branch_inputs:
                    x = branch_snapshots[self.branch_inputs[i]]
                if i in self.residuals:
                    x = self.apply_residual(x, inputs_residuals, layer=i)
                if i in self.concat_skips:
                    x = self.apply_concat(x, inputs_residuals, layer=i)
                if i in self.branch_inputs_starts:
                    branch_snapshots[i] = x
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Linear, nn.Flatten, nn.Upsample, nn.PixelShuffle)):
                    x = layer(x)
                elif isinstance(layer, nn.LayerNorm):
                    dims = tuple(range(-len(layer.normalized_shape), 0))
                    self.layernorms[i] = (torch.mean(x, dim=dims, keepdim=True), torch.var(x, dim=dims, unbiased=False, keepdim=True))
                    x = layer(x)
                elif isinstance(layer, RMSNorm):
                    self.layernorms[i] = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + layer.eps)
                    x = layer(x)
                elif isinstance(layer, nn.GroupNorm):
                    G = layer.num_groups
                    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
                    cpg = C // G
                    x_grouped = x.reshape(N, G, cpg, H, W)
                    mean = x_grouped.mean(dim=[2, 3, 4], keepdim=True)
                    var = x_grouped.var(dim=[2, 3, 4], unbiased=False, keepdim=True)
                    mean_expanded = mean.expand(-1, -1, cpg, -1, -1).reshape(N, C, 1, 1)
                    var_expanded = var.expand(-1, -1, cpg, -1, -1).reshape(N, C, 1, 1)
                    self.layernorms[i] = (mean_expanded, var_expanded)
                    x = layer(x)
                elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                    x, indices = layer(x)
                    self.maxpool_indices[i] = indices
                elif isinstance(layer, (nn.ELU, nn.LeakyReLU, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU, nn.SiLU, nn.Mish, nn.Softmax, nn.CELU, nn.Hardsigmoid, nn.Hardswish, nn.PReLU, nn.ReLU6, nn.Softplus, MultiHeadAttention)):
                    self.pre_acts[i] = x.detach().clone()
                    x = layer(x)
                    self.acts[i] = x.detach().clone()
        return x

    def apply_residual(self, x: torch.Tensor, outputs: list[torch.Tensor], layer: int, affine: bool=True) -> torch.Tensor:
        if affine:
            for start_idx, proj in self.residuals[layer]:
                output = outputs[start_idx]
                for layer in proj:
                    output = layer(output)
                x = x + output
        else:
            for start_idx, proj in self.residuals[layer]:
                output = outputs[start_idx]
                for layer in proj:
                    if isinstance(layer, nn.BatchNorm2d):
                        output = output * (layer.weight/torch.sqrt(layer.running_var+layer.eps)).view(1,-1,1,1)
                    elif isinstance(layer, nn.Linear):
                        output = torch.matmul(layer.weight, output.T).T
                    elif isinstance(layer, nn.Conv2d):
                        output = F.conv2d(output, layer.weight, None, stride=layer.stride, padding=layer.padding)
                    else:
                        output = layer(output)
                x = x + output
        return x


    ### Useful Functions ###

    def shape_at_layer(self, i: int) -> torch.Size:
        x = torch.randn(self.input_shape).unsqueeze(0)
        start_layer = self._get_start_layer()
        inputs_residuals: list[torch.Tensor] = [None] * self.get_num_layers()
        branch_snapshots: list[torch.Tensor] = [None] * self.get_num_layers()
        for j, layer in enumerate(self.layers[start_layer:i], start=start_layer):
            if j in self.concat_skips_starts:
                inputs_residuals[j] = x
            if j in self.branch_inputs:
                x = branch_snapshots[self.branch_inputs[j]]
            if j in self.concat_skips:
                x = self.apply_concat(x, inputs_residuals, layer=j)
            if j in self.branch_inputs_starts:
                branch_snapshots[j] = x
            if isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                x, _ = layer(x)
            else:
                x = layer(x)
        return x.shape

    def get_matrix_shape(self) -> Tuple[int]:
        # Returns the shape of the knowledge matrix in the format: (rows, columns).
        return (self.layers[-1].out_features, self.get_input_size() + int(self._has_bias() or self._has_batchnorm() or self._has_layernorm() or self._has_groupnorm()))
    
    def _has_bias(self) -> bool:
        for layer in self.layers:
            try: 
                _ = layer.bias.data
                return True
            except:
                continue
        return False

    def _has_batchnorm(self) -> bool:
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm2d):
                return True
        return False

    def _has_layernorm(self) -> bool:
        for layer in self.layers:
            if isinstance(layer, nn.LayerNorm):
                return True
        return False

    def _has_groupnorm(self) -> bool:
        for layer in self.layers:
            if isinstance(layer, nn.GroupNorm):
                return True
        return False

    def get_input_size(self) -> int:
        input_size = 1
        for i in self.input_shape:
            input_size *= i
        return input_size
    
    def get_num_layers(self) -> int:
        return len(self.layers)

    def eval(self) -> None:
        for layer in self.layers:
            layer.eval()
        for end in self.residuals:
            for _, proj in self.residuals[end]:
                for layer in proj:
                    layer.eval()

    def train(self) -> None:
        for layer in self.layers:
            layer.train()
        for end in self.residuals:
            for _, proj in self.residuals[end]:
                for layer in proj:
                    layer.train()

    def freeze(self) -> None:
        # Puts requires_grad = False to all parameters of all layers
        self._freeze_or_unfreeze(freeze=True)
    
    def freeze_at_layer(self, layer: int) -> None:
        # Puts requires_grad = False to all parameters of the specified layer
        for param in self.layers[layer].parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        # Puts requires_grad = True to all parameters of all layers
        self._freeze_or_unfreeze(freeze=False)
    
    def unfreeze_at_layer(self, layer: int) -> None:
        # Puts requires_grad = True to all parameters of the specified layer
        for param in self.layers[layer].parameters():
            param.requires_grad = True

    def _freeze_or_unfreeze(self, freeze: bool=True) -> None:
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = not freeze
        for end in self.residuals:
            for _, proj in self.residuals[end]:
                for layer in proj:
                    for param in layer.parameters():
                        param.requires_grad = not freeze

    def _get_start_layer(self) -> int:
        start_layer = 0
        if isinstance(self.layers[0], nn.Embedding):
            start_layer = 1
            if isinstance(self.layers[1], PositionalEncoding):
                start_layer = 2
        return start_layer


### Helpers for NN.from_torch (torch.fx graph analysis) ###

_FT_ADD_FUNCS = {operator.add, operator.iadd, torch.add}


def _ft_is_add(node: fx.Node) -> bool:
    """True if ``node`` is a tensor-add (candidate residual merge)."""
    if node.op == "call_function" and node.target in _FT_ADD_FUNCS:
        return True
    if node.op == "call_method" and node.target == "add":
        return True
    return False


def _ft_tensor_inputs(node: fx.Node) -> List[fx.Node]:
    """The fx.Node tensor producers consumed by ``node`` (args then kwargs)."""
    out = [a for a in node.args if isinstance(a, fx.Node)]
    out += [v for v in node.kwargs.values() if isinstance(v, fx.Node)]
    return out


def _ft_ancestors(node: fx.Node) -> set:
    """All transitive tensor-producing ancestors of ``node`` (excluding it)."""
    seen: set = set()
    stack = [node]
    while stack:
        n = stack.pop()
        for inp in _ft_tensor_inputs(n):
            if inp not in seen:
                seen.add(inp)
                stack.append(inp)
    return seen


def _ft_fork(a: fx.Node, b: fx.Node) -> Optional[fx.Node]:
    """
        Lowest common ancestor of two add-inputs: the tensor where the main
        and skip branches diverge. Found by walking back from ``b`` (closest
        first) until reaching a node that is also an ancestor of ``a``.
    """
    anc_a = _ft_ancestors(a)
    anc_a.add(a)
    dq = deque([b])
    visited: set = set()
    while dq:
        n = dq.popleft()
        if n in anc_a:
            return n
        if n in visited:
            continue
        visited.add(n)
        for inp in _ft_tensor_inputs(n):
            dq.append(inp)
    return None


def _ft_dist(node: fx.Node, target: fx.Node) -> Optional[int]:
    """Shortest edge distance from ``node`` back to ``target`` (0 if equal)."""
    dq = deque([(node, 0)])
    visited: set = set()
    while dq:
        n, d = dq.popleft()
        if n is target:
            return d
        if n in visited:
            continue
        visited.add(n)
        for inp in _ft_tensor_inputs(n):
            dq.append((inp, d + 1))
    return None


def _ft_linear_path(skip_arg: fx.Node, fork: fx.Node, merge: fx.Node) -> List[fx.Node]:
    """
        Ordered list of nodes on the skip branch from just-after ``fork`` to
        ``skip_arg`` (inclusive), i.e. the downsample/projection chain. Empty
        for an identity skip (``skip_arg is fork``). Requires the chain to be
        linear (single tensor input per hop); otherwise raises.
    """
    path: List[fx.Node] = []
    cur = skip_arg
    while cur is not fork:
        path.append(cur)
        ins = _ft_tensor_inputs(cur)
        if len(ins) != 1:
            raise NotImplementedError(
                f"from_torch: non-linear residual skip branch feeding node "
                f"{merge.name}; out of MVP scope."
            )
        cur = ins[0]
    path.reverse()
    return path


def _ft_arg(node: fx.Node, idx: int, key: str, default=None):
    """Read a positional-or-keyword fx call argument with a fallback."""
    if key in node.kwargs:
        return node.kwargs[key]
    if idx < len(node.args):
        return node.args[idx]
    return default


class RMSNorm(nn.Module):
    """
        Root Mean Square Layer Normalization.
        Used by LLaMA, Mistral, Gemma, Qwen, and most post-2023 LLMs.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x * self.weight / rms
   

class JumpReLU(nn.ReLU):
    """
        JumpReLU activation: z * 1[z > threshold], with per-feature thresholds.
    """
    def __init__(self, threshold: torch.Tensor):
        super().__init__()
        self.register_buffer("threshold", threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (x > self.threshold).float()


class TopKActivation(nn.ReLU):
    """
        TopK activation: keeps only the top-k activations, zeros the rest.
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk_vals, topk_idx = torch.topk(x, self.k, dim=-1)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_idx, topk_vals)
        return result


class PositionalEncoding(nn.Module):
    """
        Positional encoding for the transformer model.

        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input.
        
        Inspired by: https://medium.com/data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
        which uses the positional encoding from the Attention is All You Need paper.
    """
    def __init__(self, d_model: int, max_len: int=5000) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even.")

        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """
        Multi-head attention for the transformer model. 

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): The number of heads.
            mask (torch.Tensor): The mask to apply to the attention scores.
        
        Inspired by: https://medium.com/data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
        which is inspired by the Attention is All You Need paper.
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            num_kv_heads: Union[int, None]=None,
            mask: Union[torch.Tensor, None]=None
        ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive.")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_head = d_model // num_heads
        self.kv_repeat = num_heads // num_kv_heads
        self.mask = mask

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, num_kv_heads * self.d_head)
        self.V = nn.Linear(d_model, num_kv_heads * self.d_head)
        self.O = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, B, T, D = x.shape

        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        Q = Q.view(batch, B, T, self.num_heads, self.d_head).transpose(2, 3)
        K = K.view(batch, B, T, self.num_kv_heads, self.d_head).transpose(2, 3)
        V = V.view(batch, B, T, self.num_kv_heads, self.d_head).transpose(2, 3)

        if self.kv_repeat > 1:
            K = K.repeat_interleave(self.kv_repeat, dim=-3)
            V = V.repeat_interleave(self.kv_repeat, dim=-3)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(2, 3).contiguous().view(batch, B, T, D)
        return self.O(out)

    def eval(self) -> None:
        self.Q.eval()
        self.K.eval()
        self.V.eval()
        self.O.eval()

    def train(self) -> None:
        self.Q.train()
        self.K.train()
        self.V.train()
        self.O.train()

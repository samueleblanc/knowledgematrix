from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Union, Dict, Tuple


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
        if not self.save:  # Regular forward pass
            layers = self.layers[:-1] if return_penultimate else self.layers
            for i, layer in enumerate(layers[start_layer:], start=start_layer):
                if i in self.residuals_starts or i in self.concat_skips_starts:
                    inputs_residuals[i] = x
                if i in self.residuals:
                    x = self.apply_residual(x, inputs_residuals, layer=i)
                if i in self.concat_skips:
                    x = self.apply_concat(x, inputs_residuals, layer=i)
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
                if i in self.residuals:
                    x = self.apply_residual(x, inputs_residuals, layer=i)
                if i in self.concat_skips:
                    x = self.apply_concat(x, inputs_residuals, layer=i)
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
        for j, layer in enumerate(self.layers[start_layer:i], start=start_layer):
            if j in self.concat_skips_starts:
                inputs_residuals[j] = x
            if j in self.concat_skips:
                x = self.apply_concat(x, inputs_residuals, layer=j)
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

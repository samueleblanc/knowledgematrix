import torch
from torch import nn
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
        self.residuals: Dict[int, int] = {}


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
            bias: bool=False
        ) -> None:
        self.layers.append(nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
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
    
    def adaptiveavgpool(
            self,
            output_size: int
    ) -> None:
        self.layers.append(nn.AdaptiveAvgPool2d(output_size=output_size))
    
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
    
    def adaptivemaxpool(
            self,
            output_size: int
    ) -> None:
        self.layers.append(nn.AdaptiveMaxPool2d(output_size=output_size, return_indices=True))


    ### Dropout ###

    def dropout(self, p: float=0.5) -> None:
        self.layers.append(nn.Dropout(p=p))


    ### Activation Functions ###

    def elu(self, alpha: float=1) -> None:
        self.layers.append(nn.ELU(alpha=alpha))
    
    def leakyrelu(self, negative_slope: float=0.01) -> None:
        self.layers.append(nn.LeakyReLU(negative_slope=negative_slope))

    def relu(self) -> None:
        self.layers.append(nn.ReLU())
    
    def sigmoid(self) -> None:
        self.layers.append(nn.Sigmoid())

    def tanh(self) -> None:
        self.layers.append(nn.Tanh())


    ### Residual Connections ###

    def residual(self, start: int, end: int) -> None:
        shape_start = self.shape_at_layer(start)
        shape_end = self.shape_at_layer(end)
        if (start < end):
            if shape_start == shape_end:
                projection = nn.Identity()
            else:
                if len(shape_start) == len(shape_end):
                    if len(shape_start) >= 4 and len(shape_end) >= 4:  # Conv
                        projection = nn.Conv2d(
                            shape_start[1], 
                            shape_end[1], 
                            kernel_size=1,
                            stride = (
                                round(shape_start[2] / shape_end[2]),
                                round(shape_start[3] / shape_end[3]),
                            )
                        )
                    elif len(shape_start) <= 3 and len(shape_end) <= 3:  # FC
                        projection = nn.Linear(
                            shape_start[-1],
                            shape_end[-1]
                        )
                else:
                    raise ValueError(f"The lenghts of shape at layer {start} and {end} need to be equal to have a residual connection. Got {shape_start} and {shape_end}.")
        else:
            raise ValueError(f"To have a residual connection from layer {start} to {end}, one needs {start} < {end}.")
        
        if end in self.residuals:
                self.residuals[end].append((start, projection))
        else:
            self.residuals[end] = [(start, projection)]


    ### Forward Method ###

    def forward(self, x: torch.Tensor, return_penultimate:bool=False) -> torch.Tensor:
        x = x.unsqueeze(0)
        outputs = [x]
        if not self.save:  # Regular forward pass
            layers = self.layers[:-1] if return_penultimate else self.layers
            for i, layer in enumerate(layers):
                if isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                    x, _ = layer(x)
                else:
                    x = layer(x)
                if i in self.residuals:
                    for start_idx, proj in self.residuals[i]:
                        x = x + proj(outputs[start_idx])
                outputs.append(x)

        else:  # Forward pass for matrix computation
               # Save activations and preactivations
            self.pre_acts: list[torch.Tensor] = []
            self.acts: list[torch.Tensor] = []

            for i, layer in enumerate(self.layers):
                if isinstance(layer, nn.Conv2d):
                    x = layer(x)
                    self.pre_acts.append(x.detach().clone())
                elif isinstance(layer, nn.BatchNorm2d):
                    self.acts.append(x.detach().clone())
                    x = layer(x)
                    self.pre_acts.append(x.detach().clone())
                elif isinstance(layer, (nn.ELU, nn.LeakyReLU, nn.ReLU, nn.Sigmoid, nn.Tanh)):
                    x = layer(x)
                    self.acts.append(x.detach().clone())
                elif isinstance(layer, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                    x = layer(x)
                    self.acts.append(x.detach().clone())
                    self.pre_acts.append(x.detach().clone())
                elif isinstance(layer, (nn.AdaptiveMaxPool2d, nn.MaxPool2d)):
                    x, indices = layer(x)
                    self.acts.append(indices)
                    self.pre_acts.append(indices)
                elif isinstance(layer, nn.Linear):
                    x = layer(x)
                    if i < len(self.layers) - 1:
                        self.pre_acts.append(x.detach().clone())
                elif isinstance(layer, nn.Flatten):
                    x = layer(x)

                if i in self.residuals:
                    for start_idx, proj in self.residuals[i]:
                        x = x + proj(outputs[start_idx])
                
                outputs.append(x.detach().clone())

        return x


    ### Useful Functions ###

    def shape_at_layer(self, i: int) -> torch.Size:
        x = torch.randn(self.input_shape).unsqueeze(0)
        for layer in self.layers[:i]:
            if isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                x, _ = layer(x)
            else:
                x = layer(x)
        return x.shape

    def get_matrix_shape(self) -> Tuple[int]:
        # Returns the shape of the knowledge matrix in the format: (rows, columns).
        return (self.layers[-1].out_features, self.get_input_size() + int(self._has_bias() or self._has_batchnorm()))
    
    def _has_bias(self) -> bool:
        has_bias = False
        for layer in self.layers:
            try: 
                _ = layer.bias
                has_bias = True
                break
            except:
                continue
        return has_bias

    def _has_batchnorm(self) -> bool:
        has_batchnorm = False
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm2d):
                has_batchnorm = True
                break
        return has_batchnorm

    def get_input_size(self) -> int:
        input_size = 1
        for i in self.input_shape:
            input_size *= i
        return input_size
    
    def get_num_layers(self) -> int:
        return len(self.layers)

import torch
from torch import nn
from typing import Union
from torch.nn import functional as F

from knowledgematrix.neural_net import NN, MultiHeadAttention


class KnowledgeMatrixComputer:
    """
        A class to compute the knowledge matrix of a neural network.

        Args:
            model (NN): The neural network to compute the knowledge matrix of.
            batch_size (int): The batch size to use when computing the knowledge matrix.
            device (Union[str, None]): The device to use when computing the knowledge matrix. If None, the device of the model is used.
    """

    def __init__(
            self,
            model: NN,
            batch_size:int = 1,
            device:Union[str, None] = None
        ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.layers = model.layers
        self.device = device if device is not None else model.device
        self.in_c, self.in_h, self.in_w = model.input_shape
        self.input_size = self.in_c*self.in_h*self.in_w

        # Saves the output of the NN on the current sample in the forward method
        self.current_output: Union[NN, None] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Computes the knowledge matrix of a NN at a given input point.
            Args:
                x (torch.Tensor): The input to the NN
            Returns:
                torch.Tensor: The knowledge matrix of the NN at the input point
        """
        with torch.no_grad():
            # Saves activations and pre-activations
            self.model.save = True
            self.current_output = self.model.forward(x)
            self.model.save = False
            self.model.to(self.device)
            start_layer = self.model._get_start_layer()
            for layer in self.model.layers[:start_layer]:
                x = layer(x)
            if start_layer > 0:
                C, H, W = x.shape[1], x.shape[2], x.shape[3]
            else:
                C, H, W = x.shape[0], x.shape[1], x.shape[2]

            # Total number of positions and batches needed
            total_positions = C*H*W
            num_batches = (total_positions + self.batch_size - 1)//self.batch_size

            IN_2D = (W > 1)  # Wether the input is of shape (C,H,W) or (C,L,1)

            A = torch.Tensor().to(self.device)  # Will become the matrix M(W,f)(x)

            for batch in range(num_batches):
                # Compute batch indices
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, total_positions)
                current_batch_size = end - start

                # Create indices for this batch
                indices = torch.arange(start, end, device=self.device)
                c = indices // (H*W)
                remaining = indices % (H*W)
                h = remaining // W
                w = remaining % W

                # Create batched input for this chunk
                batched_input = torch.zeros((current_batch_size,C,H,W), device=self.device)
                batched_input[torch.arange(current_batch_size),c,h,w] = x.flatten()[start:end]

                B = batched_input
                inputs_residuals = [None] * self.model.get_num_layers()
                for i, layer in enumerate(self.layers[start_layer:], start=start_layer):
                    # Process each layer type (Conv2d, AvgPool2d, Linear, BatchNorm2d, MaxPool2d, etc.)
                    # applying the appropriate transformations and handling activation ratios
                    if i in self.model.residuals_starts:
                        inputs_residuals[i] = B.detach().clone()
                    if i in self.model.residuals:
                        B = self.model.apply_residual(B, inputs_residuals, layer=i, affine=False)
                    if isinstance(layer, (nn.ELU, nn.LeakyReLU, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU, nn.SiLU, nn.Mish, nn.Softmax, MultiHeadAttention)):
                        # Get activation ratios
                        pre_act = self.model.pre_acts[i]
                        post_act = self.model.acts[i]
                        vertices = post_act / pre_act
                        vertices = torch.where(
                            torch.isnan(vertices) | torch.isinf(vertices),
                            torch.tensor(0.0, device=self.device),
                            vertices
                        ).squeeze(0)  # Remove original batch dim
                        B = B * vertices
                    elif isinstance(layer, nn.Conv2d):
                        B = F.conv2d(B, layer.weight, None, stride=layer.stride, padding=layer.padding)
                    elif isinstance(layer, (nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Flatten)):
                        B = layer(B)
                    elif isinstance(layer, nn.Linear):
                        B = (layer.weight @ B.transpose(-1,-2)).transpose(-1,-2)
                    elif isinstance(layer, nn.BatchNorm2d):
                        B = B * (layer.weight/torch.sqrt(layer.running_var+layer.eps)).view(1,-1,1,1)
                    elif isinstance(layer, nn.LayerNorm):
                        B = B * layer.weight/torch.sqrt(self.model.layernorms[i][1]+layer.eps)
                    elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                        pool = self.model.maxpool_indices[i]
                        batch_indices = torch.arange(current_batch_size).view(-1,1,1,1)
                        channel_indices = torch.arange(pool.shape[1]).view(1,-1,1,1)
                        row_indices = pool // B.shape[2] if IN_2D else pool
                        col_indices = pool % B.shape[3]
                        B = B[batch_indices, channel_indices, row_indices, col_indices]

                B = B.reshape(-1, self.current_output.reshape(1,-1).shape[1])
                # Cat the vector produced to the matrix M(W,f)(x)
                A = torch.cat((A,B.T),dim=-1) if A.numel() else B.T

            # Process bias and batch norm terms by iterating through layers again
            # Computing activation ratios and applying appropriate transformations
            if self.model._has_bias() or self.model._has_batchnorm() or self.model._has_layernorm() or len(self.model.residuals) > 0:
                a = torch.zeros(x.shape)
                if len(x.shape) == 3:
                    a = a.unsqueeze(0)
                a = a.to(self.device)
                inputs_residuals = [None] * self.model.get_num_layers()
                for i, layer in enumerate(self.layers[start_layer:], start=start_layer):
                    if i in self.model.residuals_starts:
                        inputs_residuals[i] = a.detach().clone()
                    if i in self.model.residuals:
                        a = self.model.apply_residual(a, inputs_residuals, layer=i)
                    if isinstance(layer, (nn.ELU, nn.LeakyReLU, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU, nn.SiLU, nn.Mish, nn.Softmax, MultiHeadAttention)):
                        pre_act = self.model.pre_acts[i]
                        post_act = self.model.acts[i]
                        vertices = post_act / pre_act
                        vertices = torch.where(
                            torch.isnan(vertices) | torch.isinf(vertices),
                            torch.tensor(0.0, device=self.device),
                            vertices
                        )
                        a = a * vertices
                    elif isinstance(layer, (nn.Conv2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.BatchNorm2d, nn.Flatten, nn.Linear)):
                        a = layer(a)
                    elif isinstance(layer, nn.LayerNorm):
                        a = ((a - self.model.layernorms[i][0])/torch.sqrt(self.model.layernorms[i][1]+layer.eps))*layer.weight + layer.bias
                    elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                        pool = self.model.maxpool_indices[i]
                        batch_indices = torch.arange(pool.shape[0]).view(-1,1,1,1)
                        channel_indices = torch.arange(pool.shape[1]).view(1,-1,1,1)
                        row_indices = pool // a.shape[2] if IN_2D else pool
                        col_indices = pool % a.shape[3]
                        a = a[batch_indices, channel_indices, row_indices, col_indices]

                a = a.reshape(-1, self.current_output.reshape(1,-1).shape[1])
                return torch.cat((A, a.T), dim=-1)
            
            return A

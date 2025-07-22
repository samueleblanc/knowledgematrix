import torch
from torch import nn
from typing import Union

from knowledgematrix.neural_net import NN


class KnowledgeMatrixComputer:
    """
        A class to compute the knowledge matrix of a neural network.

        Args:
            model (NN): The neural network to compute the knowledge matrix of.
            batch_size (int): The batch size to use when computing the knowledge matrix.
    """

    def __init__(
            self,
            model: NN,
            batch_size:int = 1
        ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.layers = model.layers
        self.device = model.device
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

            # Total number of positions and batches needed
            C, H, W = self.in_c, self.in_h, self.in_w
            total_positions = C*H*W
            num_batches = (total_positions + self.batch_size - 1)//self.batch_size

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
                i = 0
                max_pool = -9
                for layer in self.layers:
                    # Get activation ratios
                    pre_act = self.model.pre_acts[i-1]
                    post_act = self.model.acts[i-1]
                    vertices = post_act / pre_act
                    vertices = torch.where(
                        torch.isnan(vertices) | torch.isinf(vertices),
                        torch.tensor(0.0, device=self.device),
                        vertices
                    ).squeeze(0)  # Remove original batch dim

                    # Process each layer type (Conv2d, AvgPool2d, Linear, BatchNorm2d, MaxPool2d)
                    # applying the appropriate transformations and handling activation ratios
                    if isinstance(layer, nn.Conv2d):
                        if i != 0 and max_pool != i-1: B = B * vertices.repeat(current_batch_size,1,1,1)
                        B = layer(B)
                        i += 1
                    elif isinstance(layer, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                        if max_pool != i-1: B = B * vertices
                        B = layer(B)
                        i += 1
                    elif isinstance(layer, nn.Linear):
                        if i != 0 and max_pool != i-1: B = B * vertices.view(1,-1).repeat(current_batch_size,1)
                        B = torch.matmul(layer.weight.data, B.T).T
                        i += 1
                    elif isinstance(layer, nn.BatchNorm2d):
                        if max_pool != i-1: B = B * vertices.repeat(current_batch_size,1,1,1)
                        B = B * (layer.weight.data/torch.sqrt(layer.running_var+layer.eps)).view(1,-1,1,1)
                        i += 1
                    elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                        max_pool = i
                        B = B * vertices.repeat(current_batch_size,1,1,1)
                        pool = self.model.acts[i]
                        batch_indices = torch.arange(current_batch_size).view(-1,1,1,1)
                        channel_indices = torch.arange(pool.shape[1]).view(1,-1,1,1)
                        row_indices = pool // B.shape[2]
                        col_indices = pool % B.shape[3]
                        B = B[batch_indices, channel_indices, row_indices, col_indices]
                        i += 1
                    elif isinstance(layer, nn.Flatten):
                        B = layer(B)

                # Cat the vector produced to the matrix M(W,f)(x)
                A = torch.cat((A,B.T),dim=-1) if A.numel() else B.T

            # Process bias and batch norm terms by iterating through layers again
            # Computing activation ratios and applying appropriate transformations
            if self.model._has_bias() or self.model._has_batchnorm():
                a = torch.zeros(x.shape).to(self.device)
                start = True
                i = 0
                max_pool = -9
                for layer in self.layers:
                    pre_act = self.model.pre_acts[i-1]
                    post_act = self.model.acts[i-1]
                    vertices = post_act / pre_act
                    vertices = torch.where(
                        torch.isnan(vertices) | torch.isinf(vertices),
                        torch.tensor(0.0, device=self.device),
                        vertices
                    )
                    if isinstance(layer, (nn.Conv2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                        if (not start) and i != 0 and max_pool != i-1:
                            a = a * vertices
                        a = layer(a)
                        i += 1
                    elif isinstance(layer, nn.Linear):
                        if (not start) and i != 0 and max_pool != i-1:
                            a = a * vertices.view(1,-1)
                        a = torch.matmul(layer.weight.data, a.T).T
                        if self.model._has_bias(): a = a + layer.bias.data.unsqueeze(0)
                        i += 1
                    elif isinstance(layer, nn.BatchNorm2d):
                        a = a * vertices
                        a = layer(a)
                        i += 1
                    elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                        max_pool = i
                        a = a * vertices
                        pool = self.model.acts[i]
                        batch_indices = torch.arange(pool.shape[0]).view(-1,1,1,1)
                        channel_indices = torch.arange(pool.shape[1]).view(1,-1,1,1)
                        row_indices = pool // a.shape[2]
                        col_indices = pool % a.shape[3]
                        a = a[batch_indices, channel_indices, row_indices, col_indices]
                        i += 1
                    elif isinstance(layer, nn.Flatten):
                        a = layer(a)
                    
                    start = False

                return torch.cat((A, a.T), dim=-1)
            
            return A

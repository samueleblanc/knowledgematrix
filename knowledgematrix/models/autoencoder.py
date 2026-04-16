import torch
from torch import nn
from typing import Tuple, Union

from knowledgematrix.neural_net import NN


class Autoencoder(NN):
    """
    Base class for autoencoders. Tracks where the encoder ends and decoder begins,
    and can extract each as standalone NN instances for knowledge matrix computation.

    Args:
        input_shape (Tuple[int]): Shape of the input (C, H, W).
        latent_shape (Tuple[int]): Shape of the latent representation (C, H, W).
        save (bool): Whether to save activations for knowledge matrix computation.
        device (str): Device to run the network on.
    """

    def __init__(
            self,
            input_shape: Tuple[int],
            latent_shape: Tuple[int],
            save: bool = False,
            device: str = "cpu"
    ) -> None:
        super().__init__(input_shape, save, device)
        self.latent_shape = latent_shape
        self.encoder_end_layer: Union[int, None] = None

    def mark_encoder_end(self) -> None:
        """Call this after adding all encoder layers to record the split point."""
        self.encoder_end_layer = self.get_num_layers()

    def get_encoder(self) -> NN:
        """Return a new NN instance containing only the encoder layers."""
        if self.encoder_end_layer is None:
            raise ValueError("encoder_end_layer not set. Call mark_encoder_end() after adding encoder layers.")
        encoder = NN(input_shape=self.input_shape, save=False, device=self.device)
        for layer in self.layers[:self.encoder_end_layer]:
            encoder.layers.append(layer)
        return encoder

    def get_decoder(self) -> NN:
        """Return a new NN instance containing only the decoder layers.

        Prepends a Flatten layer so the standalone decoder can accept
        input_shape=(d_hidden, 1, 1) from KnowledgeMatrixComputer.
        """
        if self.encoder_end_layer is None:
            raise ValueError("encoder_end_layer not set. Call mark_encoder_end() after adding encoder layers.")
        decoder = NN(input_shape=self.latent_shape, save=False, device=self.device)
        decoder.layers.append(nn.Flatten(start_dim=1, end_dim=-1))
        for layer in self.layers[self.encoder_end_layer:]:
            decoder.layers.append(layer)
        return decoder

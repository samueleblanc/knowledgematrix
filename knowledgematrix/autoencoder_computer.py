import torch
from typing import Tuple, Union

from knowledgematrix.matrix_computer import KnowledgeMatrixComputer
from knowledgematrix.models.autoencoder import Autoencoder


class AutoencoderMatrixComputer:
    """
    Computes knowledge matrices for an autoencoder's encoder, decoder, and full network.

    Args:
        model (Autoencoder): An Autoencoder instance with encoder_end_layer set.
        batch_size (int): Batch size for KnowledgeMatrixComputer.
        device (Union[str, None]): Device for computation. If None, uses model's device.
    """

    def __init__(
            self,
            model: Autoencoder,
            batch_size: int = 1,
            device: Union[str, None] = None
    ) -> None:
        self.model = model
        self.device = device or model.device

        self.encoder = model.get_encoder()
        self.decoder = model.get_decoder()

        self.encoder_computer = KnowledgeMatrixComputer(self.encoder, batch_size, device)
        self.decoder_computer = KnowledgeMatrixComputer(self.decoder, batch_size, device)
        self.full_computer = KnowledgeMatrixComputer(model, batch_size, device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute encoder, decoder, and total knowledge matrices.

        Args:
            x (torch.Tensor): Input tensor of shape input_shape.

        Returns:
            encoder_mat (torch.Tensor): Knowledge matrix of the encoder.
            decoder_mat (torch.Tensor): Knowledge matrix of the decoder.
            total_mat (torch.Tensor): Knowledge matrix of the full autoencoder.
        """
        encoder_mat = self.encoder_computer.forward(x)

        # Get latent representation for decoder input
        self.encoder.save = False
        self.encoder.eval()
        with torch.no_grad():
            latent = self.encoder.forward(x)
        latent = latent.reshape(self.model.latent_shape)

        decoder_mat = self.decoder_computer.forward(latent)

        total_mat = self.full_computer.forward(x)

        return encoder_mat, decoder_mat, total_mat

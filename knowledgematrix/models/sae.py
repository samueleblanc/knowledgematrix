import torch
from torch import nn
from typing import Union

from knowledgematrix.neural_net import JumpReLU, TopKActivation
from knowledgematrix.models.autoencoder import Autoencoder


class SAE(Autoencoder):
    """
    Sparse Autoencoder model.

    Architecture:
        Encoder: Flatten -> Linear(d_model, d_hidden) -> Activation
        Decoder: Linear(d_hidden, d_model)

    The SAE formula is:
        f(x) = sigma(W_enc @ (x - b_dec) + b_enc)
        x_hat = W_dec @ f(x) + b_dec

    When loading weights, the b_dec subtraction is folded into the encoder bias:
        encoder bias = b_enc - W_enc @ b_dec

    Args:
        d_model (int): Input/output dimension (e.g., residual stream width).
        d_hidden (int): Hidden dimension (dictionary size).
        activation (str): Activation type: 'relu', 'jumprelu', or 'topk'.
        k (int): For 'topk' activation, number of features to keep.
        thresholds (torch.Tensor): For 'jumprelu', per-feature thresholds of shape (d_hidden,).
        save (bool): Whether to save activations for knowledge matrix computation.
        device (str): Device to run the network on.
    """

    def __init__(
            self,
            d_model: int,
            d_hidden: int,
            activation: str = "relu",
            k: Union[int, None] = None,
            thresholds: Union[torch.Tensor, None] = None,
            save: bool = False,
            device: str = "cpu"
    ) -> None:
        input_shape = (d_model, 1, 1)
        latent_shape = (d_hidden, 1, 1)
        super().__init__(input_shape, latent_shape, save, device)

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.activation_type = activation

        # Encoder
        self.flatten()
        self.linear(d_model, d_hidden, bias=True)
        if activation == "relu":
            self.relu()
        elif activation == "jumprelu":
            if thresholds is None:
                thresholds = torch.zeros(d_hidden)
            self.jumprelu(thresholds)
        elif activation == "topk":
            if k is None:
                raise ValueError("k must be specified for 'topk' activation.")
            self.topk_activation(k)
        else:
            raise ValueError(f"Unknown activation: {activation}. Use 'relu', 'jumprelu', or 'topk'.")
        self.mark_encoder_end()

        # Decoder
        self.linear(d_hidden, d_model, bias=True)

    def load_weights(
            self,
            W_enc: torch.Tensor,
            b_enc: torch.Tensor,
            W_dec: torch.Tensor,
            b_dec: torch.Tensor,
            thresholds: Union[torch.Tensor, None] = None
    ) -> None:
        """
        Load SAE weights with b_dec folding.

        SAE formula:  f(x) = sigma(W_enc @ (x - b_dec) + b_enc)
        Equivalent:   f(x) = sigma(W_enc @ x + (b_enc - W_enc @ b_dec))

        So encoder Linear gets weight=W_enc, bias=(b_enc - W_enc @ b_dec)
        and decoder Linear gets weight=W_dec, bias=b_dec.

        Args:
            W_enc: Encoder weight matrix, shape (d_hidden, d_model).
            b_enc: Encoder bias, shape (d_hidden,).
            W_dec: Decoder weight matrix, shape (d_model, d_hidden).
            b_dec: Decoder bias, shape (d_model,).
            thresholds: For JumpReLU, per-feature thresholds of shape (d_hidden,).
        """
        # Fold b_dec into encoder bias
        folded_bias = b_enc - W_enc @ b_dec

        # Find the encoder Linear layer (layer after Flatten)
        encoder_linear = self.layers[1]
        encoder_linear.weight.data = W_enc.clone()
        encoder_linear.bias.data = folded_bias.clone()

        # Find the decoder Linear layer
        decoder_linear = self.layers[self.encoder_end_layer]
        decoder_linear.weight.data = W_dec.clone()
        decoder_linear.bias.data = b_dec.clone()

        # Update JumpReLU thresholds if provided
        if thresholds is not None:
            activation_layer = self.layers[2]
            if isinstance(activation_layer, JumpReLU):
                activation_layer.threshold = thresholds.clone()

    @classmethod
    def from_saelens(
            cls,
            release: str,
            sae_id: str,
            device: str = "cpu"
    ) -> "SAE":
        """
        Load a pretrained SAE from SAELens.

        Requires: pip install sae-lens

        Args:
            release (str): SAELens release name (e.g., "gpt2-small-res-jb").
            sae_id (str): SAE identifier (e.g., "blocks.8.hook_resid_post").
            device (str): Device to load the model on.

        Returns:
            SAE: A knowledge-matrix-compatible SAE with pretrained weights.
        """
        try:
            from sae_lens import SAE as SAELensSAE
        except ImportError:
            raise ImportError(
                "sae-lens is required to load pretrained SAEs. "
                "Install it with: pip install sae-lens"
            )

        sae_lens_model = SAELensSAE.from_pretrained(release, sae_id, device=device)

        d_model = sae_lens_model.cfg.d_in
        d_hidden = sae_lens_model.cfg.d_sae

        # Determine activation type and parameters
        act_type = "relu"
        k = None
        thresholds = None

        activation_fn = getattr(sae_lens_model.cfg, "activation_fn", "relu")
        if isinstance(activation_fn, str):
            act_fn_name = activation_fn.lower()
        else:
            act_fn_name = type(activation_fn).__name__.lower()

        if "jumprelu" in act_fn_name:
            act_type = "jumprelu"
            if hasattr(sae_lens_model, "threshold"):
                thresholds = sae_lens_model.threshold.data.to(device)
            else:
                thresholds = torch.zeros(d_hidden, device=device)
        elif "topk" in act_fn_name:
            act_type = "topk"
            k = getattr(sae_lens_model.cfg, "k", d_hidden // 10)

        sae = cls(
            d_model=d_model,
            d_hidden=d_hidden,
            activation=act_type,
            k=k,
            thresholds=thresholds,
            device=device
        )

        # Extract weights from SAELens model
        W_enc = sae_lens_model.W_enc.data.T.to(device)  # SAELens: (d_model, d_sae) -> (d_sae, d_model)
        b_enc = sae_lens_model.b_enc.data.to(device)
        W_dec = sae_lens_model.W_dec.data.T.to(device)  # SAELens: (d_sae, d_model) -> (d_model, d_sae)
        b_dec = sae_lens_model.b_dec.data.to(device)

        sae.load_weights(W_enc, b_enc, W_dec, b_dec, thresholds)

        return sae

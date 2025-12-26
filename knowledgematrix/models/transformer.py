"""
    Implementation of a Transformer
"""

from knowledgematrix.neural_net import NN


class Transformer(NN):
    """
        A transformer model. This is an encoder-only version of the model introduced in the Attention is All You Need paper.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feed-forward network.
            num_heads (int): The number of heads.
            max_len (int): The maximum length of the input.
            save (bool): Whether to save the activations and preactivations of the network.
            device (str): The device to run the network on.
    """
    def __init__(
            self, 
            vocab_size: int=100,
            d_model: int=100,
            d_ff: int=400,
            num_heads: int=10,
            max_len: int=1000,
            save: bool=False, 
            device: str="cpu"
        ) -> None:
        super().__init__(input_shape=(1, 1, d_model), save=save, device=device)

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.d_ff = d_ff
        self.max_len = max_len
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.embedding(vocab_size, d_model)
        self.positionalencoding(d_model, max_len)

        # Block 1
        # Multi-head attention
        start_skip = self.get_num_layers()
        self.multiheadattention(d_model, num_heads)
        self.dropout(0.5)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self.layernorm(d_model)

        # Feed-forward
        start_skip = self.get_num_layers()
        self.linear(in_features=d_model, out_features=d_ff)
        self.gelu()
        self.linear(in_features=d_ff, out_features=d_model)
        self.dropout(0.5)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self.layernorm(d_model)

        # Block 2
        # Multi-head attention
        start_skip = self.get_num_layers()
        self.multiheadattention(d_model, num_heads)
        self.dropout(0.5)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self.layernorm(d_model)

        # Feed-forward
        start_skip = self.get_num_layers()
        self.linear(in_features=d_model, out_features=d_ff)
        self.gelu()
        self.linear(in_features=d_ff, out_features=d_model)
        self.dropout(0.5)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self.layernorm(d_model)

        # Block 3
        # Multi-head attention
        start_skip = self.get_num_layers()
        self.multiheadattention(d_model, num_heads)
        self.dropout(0.5)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self.layernorm(d_model)

        # Feed-forward
        start_skip = self.get_num_layers()
        self.linear(in_features=d_model, out_features=d_ff)
        self.gelu()
        self.linear(in_features=d_ff, out_features=d_model)
        self.dropout(0.5)
        end_skip = self.get_num_layers()
        self.residual(start_skip, end_skip)
        self.layernorm(d_model)

        self.linear(in_features=d_model, out_features=vocab_size)
        self.softmax()

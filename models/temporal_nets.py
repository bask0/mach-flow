
import torch
from torch import Tensor

from models.base import TemporalNet
from utils.torch_modules import Transform, SeqZeroPad, TemporalConvNet, Transform

class TCN(TemporalNet):
    """TCN based rainfall-runoff module."""
    def __init__(
            self,
            model_dim: int = 32,
            tcn_kernel_size: int = 4,
            tcn_dropout:float = 0.0,
            tcn_layers: int = 1,
            **kwargs) -> None:

        super().__init__(model_dim=model_dim, **kwargs)

        self.save_hyperparameters()

        self.model_type = 'TCN'

        self.tcn = TemporalConvNet(
            num_inputs=model_dim,
            num_outputs=-1,
            num_hidden=model_dim,
            kernel_size=tcn_kernel_size,
            num_layers=tcn_layers,
            dropout=tcn_dropout
        )

    def temporal_forward(self, x: Tensor) -> Tensor:
        out = self.tcn(x)
        return out


class LSTM(TemporalNet):
    """LSTM based rainfall-runoff module."""
    def __init__(
            self,
            model_dim: int = 32,
            lstm_layers: int = 1,
            **kwargs) -> None:

        super().__init__(model_dim=model_dim, **kwargs)

        self.model_type = 'LSTM'

        self.save_hyperparameters()

        self.to_channel_last = Transform(transform_fun=lambda x: x.transpose(1, 2), name='Transpose(1, 2)')

        self.lstm = torch.nn.LSTM(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.to_sequence_last = Transform(transform_fun=lambda x: x.transpose(1, 2), name='Transpose(1, 2)')

    def temporal_forward(self, x: Tensor) -> Tensor:
        out = self.to_channel_last(x)
        out, _ = self.lstm(out)
        out = self.to_sequence_last(out)
        return out


class RelativePositionalEncoder(torch.nn.Module):
    """Relative positional encoding.

    # https://github.com/tensorflow/tensor2tensor
    # https://github.com/gazelle93/Attention-Various-Positional-Encoding/blob/main/positional_encoders.py

    """
    def __init__(self, emb_dim: int, max_position: int = 512) -> None:
        """Initialize RelativePositionalEncoding.

        Inputs:
            seq_len_q: The length of the query sequence.
            seq_len_k: the length of the key sequence.

        Returns:
            Relative positional embeddings with shape 

        Args:
            emb_dim: The embedding dimensionality.
            max_position: The maximum relative position that can be represented by this encoder.
        """
        super().__init__()
        self.max_position = max_position
        self.embeddings_table = torch.nn.Parameter(torch.Tensor(max_position * 2 + 1, emb_dim))
        torch.nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, seq_len_q: int, seq_len_k: int) -> Tensor:

        range_vec_q = torch.arange(seq_len_q)

        range_vec_k = torch.arange(seq_len_k)
        relative_matrix = range_vec_k[None, :] - range_vec_q[:, None]
        clipped_relative_matrix = torch.clamp(relative_matrix, -self.max_position, self.max_position)
        relative_position_matrix = clipped_relative_matrix + self.max_position

        embeddings = self.embeddings_table[relative_position_matrix]

        return embeddings


class RelativeScaledDotProductAttention(torch.nn.Module):
    """Scaled Dot Product Attention using Relative Positional Encoding

    https://github.com/gazelle93/Attention-Various-Positional-Encoding/blob/main/attentions.py

    """
    def __init__(self, emb_dim: int) -> None:
        super().__init__()

        # scaling factor 1 / sqrt(dimension of queries and keys)
        self.scaling_factor = torch.sqrt(torch.tensor(emb_dim))

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            a_key: Tensor,
            a_value: Tensor,
            mask: Tensor | None = None) -> tuple[Tensor, Tensor]:

        # Scaled score of the Matrix multiplication of query and key (e)
        qk_attn = torch.bmm(query, key.transpose(1, 2))
        relative_qk_attn = torch.bmm(query.permute(1, 0, 2).contiguous(), a_key.transpose(1, 2)).transpose(0, 1)
        attn_score = (qk_attn + relative_qk_attn) / self.scaling_factor

        # Masking (Optional)
        # shape of mask: (batch size, input length of query, input length of key)
        if mask is not None:
            attn_score.masked_fill_(mask, -1e18)

        # Softmax of the scaled score (alpha)
        attn_score = torch.softmax(attn_score, -1)

        # Matrix multiplication of the scaled score and value (z)
        qkv_attn = torch.bmm(attn_score, value)
        relative_qkv_attn = torch.bmm(attn_score.permute(1, 0, 2).contiguous(), a_value).transpose(0, 1)

        output = qkv_attn + relative_qkv_attn

        return output, attn_score


class MultiheadAttention(torch.nn.Module):
    """Multihead attention using relation positional encoding.

    https://github.com/gazelle93/Attention-Various-Positional-Encoding/blob/main/attentions.py

    """
    def __init__(self, emb_dim: int, num_heads: int, dropout: float = 0.1, max_position: int = 512) -> None:
        super().__init__()

        self.head_dim = int(emb_dim / num_heads)
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(dropout)

        # initialize one feed-forward layer (head dimension x number of heads) of each q, k and v
        # instead of initializing number of heads of feed-forward layers (head dimension / number of heads)
        self.query_proj = torch.nn.Linear(emb_dim, self.head_dim * num_heads)
        self.key_proj = torch.nn.Linear(emb_dim, self.head_dim * num_heads)
        self.value_proj = torch.nn.Linear(emb_dim, self.head_dim * num_heads)
        self.out_proj = torch.nn.Linear(emb_dim, self.head_dim * num_heads)

        self.relative_scaled_dot_attn = RelativeScaledDotProductAttention(self.head_dim)
        self.relative_position_k = RelativePositionalEncoder(self.head_dim, max_position=max_position)
        self.relative_position_v = RelativePositionalEncoder(self.head_dim, max_position=max_position)

    def reshape_from_feed_forward(self, batch_size: int, _tensor: Tensor) -> Tensor:
        return _tensor.view(batch_size, -1, self.num_heads, self.head_dim)

    def reshape_to_ScaledDotProductAttention(self, batch_size: int, _tensor: Tensor) -> Tensor:
        # before shape: (batch size, input length, number of heads, head dimension)
        # after shape: (batch size, number of heads, input length, head dimension)
        _tensor = _tensor.permute(0, 2, 1, 3)

        # reshape to feed the tensor to ScaledDotProductAttention
        return _tensor.contiguous().view(batch_size * self.num_heads, -1, self.head_dim)

    def reshape_to_concat(self, batch_size: int, _tensor: Tensor) -> Tensor:
        # before shape: (number of heads, batch size, input length, head dimension)
        # after shape: (batch size, input length, number of heads, head dimension)
        _tensor = _tensor.permute(1, 2, 0, 3)
        return _tensor.contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            mask: Tensor | None  = None,
            is_dropout: bool=True) -> tuple[Tensor, Tensor]:

        # Shape of input of q, k and v: (batch size, input length, embedding dimension).
        batch_size = query.size()[0]

        # Feed-forward network:
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # reshape the result of the feed-forward network
        # shape after the feed-forward network of q, k and v: (batch, input length, number of heads, head dimension)
        query = self.reshape_from_feed_forward(batch_size, query)
        key = self.reshape_from_feed_forward(batch_size, key)
        value = self.reshape_from_feed_forward(batch_size, value)

        # reshape the result of the feed-forward network to feed it to ScaledDotProductAttention
        # shape: (number of heads * batch, input length, head dimension)
        query = self.reshape_to_ScaledDotProductAttention(batch_size, query)
        key = self.reshape_to_ScaledDotProductAttention(batch_size, key)
        value = self.reshape_to_ScaledDotProductAttention(batch_size, value)


        # shape of mask: (batch size, number of heads, input length of query, input length of key)
        if mask is not None:
            # mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            mask = mask.unsqueeze(0)

        seq_len_query = query.size()[1]
        seq_len_key = key.size()[1]
        seq_len_value = value.size()[1]
        a_key = self.relative_position_k(seq_len_query, seq_len_key)
        a_value = self.relative_position_v(seq_len_query, seq_len_value)

        # ourput shape: (number of heads x batch size, input length, head dimension)
        output, attn_score = self.relative_scaled_dot_attn(query, key, value, a_key, a_value, mask)

        # reshape the result of the ScaledDotProductAttention
        # shape: (number of heads, batch size, input length, head dimension)
        output = output.view(self.num_heads, batch_size, -1, self.head_dim)

        # reshape to concat
        # shape: (number of heads, batch size, input length, head dimension)
        output = self.reshape_to_concat(batch_size, output)

        # final feed-forward network
        output = self.out_proj(output)
        
        if is_dropout:
            output = self.dropout(output)
            return output, attn_score
        
        return output, attn_score


class EncoderBlock(torch.nn.Module):
    """

    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """
    def __init__(
            self,
            model_dim: int,
            num_heads: int,
            dropout: float = 0.1,
            max_position:  int = 100) -> None:
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(
            emb_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_position=max_position
        )

        # Two-layer MLP
        self.linear_net = torch.nn.Sequential(
            torch.nn.Linear(model_dim, model_dim * 2),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(model_dim * 2, model_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = torch.nn.LayerNorm(model_dim)
        self.norm2 = torch.nn.LayerNorm(model_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # Attention part
        attn_out, _ = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


def generate_causal_mask(
        sz: int,
        max_context: int | None = None,
        device: torch.device | str = 'cpu') -> Tensor:
    r"""Generate a square causal mask for the sequence. The masked positions are filled with True.
        Unmasked positions are filled with False.
    """

    causal_mask = torch.triu(torch.full((sz, sz), True, device=device), diagonal=1)

    if max_context is not None:
        max_context_mask = torch.tril(
            torch.full((sz, sz), True, device=device),
            diagonal=-max_context - 1,
        )
        causal_mask += max_context_mask

    return causal_mask


class TransformerEncoderBlock(torch.nn.Module):
    """

    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """
    def __init__(
            self,
            model_dim: int,
            num_heads: int,
            num_layers: int,
            max_context: int,
            dropout: float = 0.1) -> None:
        super().__init__()

        self.max_context = max_context
        self.layers = torch.nn.ModuleList(
            [EncoderBlock(model_dim=model_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:

        # x: (batch, sequence, channels)
        seq_length = x.shape[1]

        causal_mask = generate_causal_mask(sz=seq_length, max_context=self.max_context, device=x.device)

        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        return x

    def get_attention_maps(self, x: Tensor, mask: Tensor | None = None) -> list[Tensor]:

        attention_maps = []

        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)

        return attention_maps


class MHA(TemporalNet):
    """Transformer based rainfall-runoff module."""
    def __init__(
            self,
            model_dim: int = 32,
            mha_heads: int = 4,
            mha_layers: int = 1,
            mha_locality_kernel_size: int = 4,
            mha_max_context: int = 100,
            mha_dropout: float = 0.1,
            **kwargs) -> None:

        super().__init__(model_dim=model_dim, **kwargs)

        self.model_type = 'MHA'

        self.save_hyperparameters()

        self.model_dim = model_dim
        self.max_context = mha_max_context

        self.zero_pad = SeqZeroPad(
            n_pad=mha_locality_kernel_size - 1
        )

        self.input_embedding = torch.nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=mha_locality_kernel_size
        )

        self.input_activation = torch.nn.Tanh()

        self.to_channel_last = Transform(transform_fun=lambda x: x.transpose(1, 2), name='Transpose(1, 2)')

        self.transformer_encoder = TransformerEncoderBlock(
            model_dim=model_dim,
            num_heads=mha_heads,
            num_layers=mha_layers,
            max_context=mha_max_context,
            dropout=mha_dropout
        )

        self.to_sequence_last = Transform(transform_fun=lambda x: x.transpose(1, 2), name='Transpose(1, 2)')

    def temporal_forward(self, x: Tensor) -> Tensor:

        # Locality-enhanced encoding: (B, D, S) -> (B, D, S)
        x_emb = self.input_activation(self.input_embedding(self.zero_pad(x)))

        # To channel last: (B, D, S) -> (B, S, D)
        x_emb = self.to_channel_last(x_emb)

        # Temporal layer: (B, S, D) -> (B, S, D)
        out = self.transformer_encoder(x_emb)

        # To sequence last: (B, S, D) -> (B, D, S)
        out = self.to_sequence_last(out)

        return out

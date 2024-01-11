import lightning.pytorch as pl
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
import logging
import torch
from torch import Tensor
import numpy as np
import os
import warnings
import abc

from utils.loss_functions import RegressionLoss
from utils.types import BatchPattern, ReturnPattern
from utils.torch_modules import DataTansform, EncodingModule, PadTau, Transform, SeqZeroPad


# Ignore anticipated PL warnings.
warnings.filterwarnings('ignore', '.*infer the indices fetched for your dataloader.*')
warnings.filterwarnings('ignore', '.*You requested to overfit.*')

logger = logging.getLogger('lightning')


class LightningNet(pl.LightningModule):
    """Implements basic training routine.

    Note:
        * This class should take hyperparameters for the training process. Model hyperparameters should be
            handled in the PyTorch module.
        * call 'self.save_hyperparameters()' at the end of subclass `__init__(...)`.
        * The subclass must implement a `forward` method (see PyTorch doc) which takes the arguments
            `x`, the sequencial input, and `s`, the static features.

    Shape:
        The subclass must implement a `forward` method (see PyTorch doc) which takes the arguments `x`, the
        sequencial input, and optional argument `s`, the static input:
        * `x`: (B, C, S)
        * `s`: (B, D)
        * return: (B, O, S)
        where B=batch size, S=sequence length, C=number of dynamic channels, and D=number of static channels.
    """
    def __init__(
            self,
            criterion: str = 'L2',
            sqrt_transform: bool = False,
            inference_taus: list[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
            norm_args_features: dict | None = None,
            norm_args_stat_features: dict | None = None,
            norm_args_targets: dict | None = None
            ) -> None:
        """Initialize lightning module, should be subclassed.

        Args:
            criterion: the criterion to use, defaults to L2.
            sqrt_transform: Whether to sqrt-transform the predictions and targets before computing the loss.
                Default is False.
            inference_taus: The tau values (probabilities) to evaluate in inference. Only applies for
                distribution-aware criterions.
        """

        super().__init__()

        self.loss_fn = RegressionLoss(criterion=criterion, sqrt_transform=sqrt_transform)

        self.sample_tau = self.loss_fn.has_tau
        self.inference_taus = inference_taus if self.sample_tau else [0.5]

        if norm_args_features is not None:
            self.norm_features = DataTansform(**norm_args_features)
        else:
            self.norm_features = torch.nn.Identity()

        if norm_args_stat_features is not None:
            self.norm_stat_features = DataTansform(**norm_args_stat_features)
        else:
            self.norm_stat_features = torch.nn.Identity()

        if norm_args_targets is not None:
            self.norm_targets = DataTansform(**norm_args_targets)
        else:
            self.norm_targets = torch.nn.Identity()

    def normalize_data(self, batch: BatchPattern) -> tuple[Tensor, Tensor, Tensor]:
        x = self.norm_features(batch.dfeatures)
        s = self.norm_stat_features(batch.sfeatures)
        y = self.norm_targets(batch.dtargets)

        return x, s, y

    def denormalize_target(self, target: Tensor) -> Tensor:
        if isinstance(self.norm_targets, torch.nn.Identity):
            return target
        else:
            return self.norm_targets(target, invert=True)

    def shared_step(
            self,
            batch: BatchPattern,
            step_type: str,
            tau: float = 0.5) -> tuple[Tensor, ReturnPattern]:
        """A single training step shared across specialized steps that returns the loss and the predictions.

        Args:
            batch: the bach.
            step_type: the step type (training mode), one of (`train`, `val`, `test`, `pred`).

        Returns:
            tuple[Tensor, ReturnPattern]: the loss, the predictions.
        """

        if step_type not in ('train', 'val', 'test', 'pred'):
            raise ValueError(f'`step_type` must be one of (`train`, `val`, `test`, `pred`), is {step_type}.')

        x, s, target = self.normalize_data(batch)

        target_hat = self(
            x=x,
            s=s,
            tau=tau
        )

        num_cut = batch.coords.warmup_size[0]
        batch_size = target_hat.shape[0]

        if self.loss_fn.criterion in ['quantile', 'expectile']:
            loss = self.loss_fn(
                input=target_hat[..., num_cut:],
                target=target[..., num_cut:],
                tau=tau
            )
        else:
            loss = self.loss_fn(
                input=target_hat[..., num_cut:],
                target=target[..., num_cut:],
            )

        preds = ReturnPattern(
            dtargets=self.denormalize_target(target_hat.detach()),
            coords=batch.coords,
            tau=tau
        )

        if step_type != 'pred':
            self.log_dict(
                {f'{step_type}_loss': loss},
                prog_bar=True,
                on_step=True if step_type == 'train' else False,
                on_epoch=True,
                batch_size=batch_size
            )

        return loss, preds

    def training_step(
            self,
            batch: BatchPattern,
            batch_idx: int) -> Tensor:
        """A single training step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        Returns:
            Tensor: The batch loss.
        """

        if self.sample_tau:
            tau = np.random.uniform()
        else:
            tau = 0.5

        loss, _ = self.shared_step(batch, step_type='train', tau=tau)

        return loss

    def validation_step(
            self,
            batch: BatchPattern,
            batch_idx: int) -> dict[str, Tensor]:
        """A single validation step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        """

        loss, _ = self.shared_step(batch, step_type='val', tau=0.5)

        return {'val_loss': loss}

    def test_step(
            self,
            batch: BatchPattern,
            batch_idx: int) -> dict[str, Tensor]:
        """A single test step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        """

        loss, _ = self.shared_step(batch, step_type='test', tau=0.5)

        return {'test_loss': loss}

    def predict_step(
            self,
            batch: BatchPattern,
            batch_idx: int,
            dataloader_idx: int = 0
            ) -> list[ReturnPattern]:
        """A single predict step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        """

        tau_preds = []

        for tau in self.inference_taus:
            _, preds = self.shared_step(batch, step_type='pred', tau=tau)
            tau_preds.append(preds)

        return tau_preds

    def summarize(self):
        s = f'=== Summary {"=" * 31}\n'
        s += f'{str(ModelSummary(self))}\n\n'
        s += f'=== Model {"=" * 33}\n'
        s += f'{str(self)}'

        return s

    def on_train_start(self) -> None:

        if self.logger is None:
            raise AttributeError(
                'self.logger is not set.'
            )

        if not hasattr(self.logger, 'log_dir') or (self.logger.log_dir is None):
            raise KeyError('logger has no attribute \'log_dir\', or it is None.')

        os.makedirs(self.logger.log_dir, exist_ok=True)
        with open(os.path.join(self.logger.log_dir, 'model_summary.txt'), 'w') as f:
            f.write(self.summarize())

        return super().on_train_start()


class TemporalCombine(torch.nn.Module):
    """Combine dynamic and static features.

    Shapes:
        x: The dynamic inputs, shape (batch, num_dynamic_in, sequence)
        s: The static inputs, shape (batch, num_static_in)

        output: A tensor of combined dynamic and static features, shape (batch, num_out, sequence).

    Args:
        num_dynamic_in: Number of dynamic inputs.
        num_static_in: Number of static inputs.
        num_out: Encoding size.
        hidden_size_factor: The hidden size is num_out * hidden_size_factor.
        dropout: Dropout applied after each layer but last, a float in the range [0, 1]. Default is 0.0.

    """
    def __init__(
            self,
            num_dynamic_in: int,
            num_static_in: int,
            num_out: int,
            num_layers: int,
            hidden_size_factor: int = 2,
            dropout: float = 0.0) -> None:
        """Initialize TemporalCombine layer."""
        super().__init__()

        self.encoding_layer = EncodingModule(
            num_in=num_dynamic_in + num_static_in,
            num_enc=num_out,
            num_layers=num_layers,
            dropout=dropout,
            hidden_size_factor=hidden_size_factor,
            activation=torch.nn.Tanh(),
            activation_last=torch.nn.Tanh()
        )

    def get_expand_arg(self, x: Tensor, s: Tensor) -> list[int]:
        expand_size = list(x.shape)
        expand_size[1] = s.shape[-1]

        return expand_size

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        s = s.unsqueeze(-1).expand(*self.get_expand_arg(x=x, s=s))

        xs = torch.cat((x, s), dim=1)

        return self.encoding_layer(xs)


class TemporalNet(LightningNet, abc.ABC):
    def __init__(
            self,
            num_static_in: int,
            num_dynamic_in: int,
            num_outputs: int,
            model_dim: int = 8,
            enc_dropout: float = 0.0,
            pre_fusion: bool = True,
            **kwargs) -> None:

        super().__init__(**kwargs)

        self.pre_fusion = pre_fusion

        # Input encoding
        # -------------------------------------------------

        if self.pre_fusion:

            self.pre_fusion_layer = TemporalCombine(
                num_dynamic_in=num_dynamic_in,
                num_static_in=num_static_in,
                num_out=model_dim,
                num_layers=2,
                dropout=0.1,
            )

        else:

            self.input_layer = torch.nn.Conv1d(
                in_channels=num_dynamic_in,
                out_channels=model_dim,
                kernel_size=1
            )

        self.in_dropout1d = torch.nn.Dropout1d(enc_dropout)

        # Temporal layer
        # -------------------------------------------------

        # >>>> Is defined in subclass.

        # Output
        # -------------------------------------------------

        if not self.pre_fusion:
            self.post_fusion_layer = TemporalCombine(
                    num_dynamic_in=model_dim,
                    num_static_in=num_static_in,
                    num_out=model_dim,
                    num_layers=2,
                    dropout=0.1,
                )

        self.out_dropout1d = torch.nn.Dropout1d(enc_dropout)

        self.pad_tau = PadTau()

        self.output_layer = torch.nn.Conv1d(
                in_channels=model_dim + 1,
                out_channels=num_outputs,
                kernel_size=1
        )

        self.out_activation = torch.nn.Softplus()

    """Implements a temporla base lightning network."""
    @abc.abstractmethod
    def temporal_forward(self, x: Tensor) -> Tensor:
        """Temporal layer forward pass, must be overridden in subclass.

        Shapes:
            x: (batch, channels, sequence).

        Returns:
            Tensor with same shape as input (batch, channels, sequence).

        Args:
            x: The input tensor of shape (batch, channels, sequence),

        Returns:
            A tensor of same shape as the input.
        """

    def forward(self, x: Tensor, s: Tensor, tau: float) -> Tensor:

        if self.pre_fusion:

            # Fusion of dynamic and static features: (B, D, S), (B, C) -> (B, E, S)
            enc = self.pre_fusion_layer(x=x, s=s)

        else:

            # Encoding of dynamic features: (B, D, S) -> (B, E, S)
            enc = self.input_layer(x)

        # 2D dropout.
        enc = self.in_dropout1d(enc)

        # Temporal layer: (B, E, S) -> (B, E, S)
        out = self.temporal_forward(enc)

        if not self.pre_fusion:
            # Fusion of dynamic and static features: (B, E, S), (B, C) -> (B, E, S)
            out = self.post_fusion_layer(x=out, s=s)

        # 2D dropout.
        enc = self.out_dropout1d(enc)

        # Pad tau: (B, E, S) -> (B, E + 1, S)
        out = self.pad_tau(x=out, tau=tau)

        # Output layer, and activation: (B, E + 1, S) -> (B, O, S)
        out = self.output_layer(out)
        out = self.out_activation(out)

        return out


from utils.torch_modules import TemporalConvNet, Transform

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
        # before shape: (batch size, number of heads, input length, head dimension)
        # after shape: (batch size, input length, number of heads, head dimension)
        _tensor = _tensor.permute(0, 2, 1, 3)
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
            mha_max_context: int = 200,
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


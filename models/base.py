import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary
import logging
import torch
from torch import Tensor
import numpy as np
import os
import warnings
import abc

from utils.loss_functions import RegressionLoss
from utils.types import BatchPattern, ReturnPattern
from utils.torch_modules import Normalize, EncodingModule, PadTau


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
            log_transform: bool = False,
            inference_taus: list[float] = [0.05, 0.25, 0.5, 0.75, 0.9],
            norm_args_features: dict | None = None,
            norm_args_stat_features: dict | None = None,
            norm_args_targets: dict | None = None
            ) -> None:
        """Initialize lightning module, should be subclassed.

        Args:
            criterion: the criterion to use, defaults to L2.
            log_transform: Whether to log-transform the predictions and targets before computing the loss.
                Default is False.
            inference_taus: The tau values (probabilities) to evaluate in inference. Only applies for
                distribution-aware criterions.
        """

        super().__init__()

        self.loss_fn = RegressionLoss(criterion=criterion, log_transform=log_transform)

        self.sample_tau = self.loss_fn.has_tau
        self.inference_taus = inference_taus

        if norm_args_features is not None:
            self.norm_features = Normalize(**norm_args_features)
        else:
            self.norm_features = torch.nn.Identity()

        if norm_args_stat_features is not None:
            self.norm_stat_features = Normalize(**norm_args_stat_features)
        else:
            self.norm_stat_features = torch.nn.Identity()

        if norm_args_targets is not None:
            self.norm_targets = Normalize(**norm_args_targets)
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

        os.makedirs(self.logger.log_dir, exist_ok=True)
        with open(os.path.join(self.logger.log_dir, 'model_summary.txt'), 'w') as f:
            f.write(self.summarize())

        return super().on_train_start()


class TemporalNet(LightningNet, abc.ABC):
    def __init__(
            self,
            num_static_in: int,
            num_dynamic_in: int,
            num_outputs: int,
            model_dim: int = 8,
            enc_layers: int = 1,
            enc_dropout: float = 0.0,
            **kwargs) -> None:

        super().__init__(**kwargs)

        # Static input encoding
        # -------------------------------------------------

        self.static_encoding = EncodingModule(
            num_in=num_static_in,
            num_enc=model_dim - 1,
            num_layers=enc_layers,
            dropout=enc_dropout,
            activation=torch.nn.ReLU(),
            activation_last=torch.nn.Sigmoid()
        )

        # Dynamic input encoding
        # -------------------------------------------------

        self.dynamic_encoding = EncodingModule(
            num_in=num_dynamic_in,
            num_enc=model_dim - 1,
            num_layers=enc_layers,
            dropout=enc_dropout,
            activation=torch.nn.Tanh(),
            activation_last=torch.nn.Tanh()
        )

        # Output
        # -------------------------------------------------

        self.pad_tau = PadTau()

        self.out_layer = torch.nn.Conv1d(
            in_channels=model_dim + 1,
            out_channels=num_outputs,
            kernel_size=1
        )

        self.out_activation = torch.nn.Softplus()

    """Implements a temporla base lightning network."""
    @abc.abstractmethod
    def temporal_forward(self, x: Tensor) -> Tensor:
        """Temporal layer forward pass, must be overridden in subclass.

        Args:
            x: The input tensor of shape (batch, channels, sequence),

        Returns:
            A tensor of same shape as the input.
        """

    def encode(self, x: Tensor, s: Tensor | None) -> Tensor:
        """Encode dynamic and static features.

        Args:
            x: The dynamic inputs, shape (batch, dynamic_channels, sequence)
            x: The static inputs, shape (batch, static_channels)

        Returns:
            A tensor of combined dynamic and static features, shape (batch, encoding_dim, sequence).
        """
        # Dynamic encoding: (B, D, S) -> (B, E, S)
        x_enc = self.dynamic_encoding(x)

        if s is not None:
            # Static encoding and unsqueezing: (B, C) ->  (B, E, 1)
            s_enc = self.static_encoding(s.unsqueeze(-1))

            # Add static encoding to dynamic encoding: (B, E, S) + (B, E, 1) -> (B, E, S)
            x_enc = x_enc + s_enc

        return x_enc

    def forward(self, x: Tensor, s: Tensor, tau: float) -> Tensor:

        # Encoding of dynamic and static features: (B, D, S), (B, C) -> (B, E - 1, S)
        enc = self.encode(x=x, s=s)

        # Pad tau: (B, E - 1, S) -> (B, E, S)
        enc = self.pad_tau(enc, tau)

        # Temporal layer: (B, E, S) -> (B, E, S)
        out = self.temporal_forward(enc)

        # Pad tau: (B, E, S) -> (B, E + 1, S)
        out = self.pad_tau(out, tau)

        # Output layer and activation: (B, E + 1, S) -> (B, O, S)
        out = self.out_layer(out)
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
        self.model_type = 'TCN'

        self.save_hyperparameters()

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
    """TCN based rainfall-runoff module."""
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



import math

class PositionalEncoding(torch.nn.Module):
    """Positional encoding as in Attention is all you need https://arxiv.org/pdf/1706.03762.pdf

    Adapted from the PyTorch tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(
            self,
            d_model: int,
            max_len: int,
            num_unique: int = 10000,
            dropout: float = 0.1):
        """Initialize PositionalEncoging.

        Args:
            d_model: Model dimensionalitz / embedding size.
            max_len: Maximum length that needs to be encoded.
            num_unique: Number of unique values that can be encoded. Should be >= `max_len`. Defaults to 10k.
            dropout: The dropout applied after positional encoding. Default is 0.1.

        """
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.num_unique = num_unique

        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(num_unique) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.pe: Tensor

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [S, N, E].

        Returns:
            Tensor, shape [S, N, E].
        """

        out = x + self.pe[:x.size(0)]
        return self.dropout(out)

    def __repr__(self) -> str:
        s = f'PositionalEncoding(d_model={self.d_model}, max_len={self.max_len}, num_unique={self.num_unique}, '
        s += f'dropout={self.dropout}): [S, N, E] -> [S, N, E]'
        return s



import math

class MHA(TemporalNet):
    """TCN based rainfall-runoff module."""
    def __init__(
            self,
            model_dim: int = 32,
            mha_nhead: int = 4,
            mha_layers: int = 1,
            mha_dropout: float = 0.1,
            mhs_max_context: int = 365,
            mhs_pos_enc_length: int = 1000,
            **kwargs) -> None:

        super().__init__(model_dim=model_dim, **kwargs)
        self.model_type = 'Transformer'

        self.save_hyperparameters()

        self.model_dim = model_dim
        self.max_context = mhs_max_context

        self.pos_encoder = PositionalEncoding(
            d_model=model_dim,
            max_len=mhs_pos_enc_length,
            num_unique=1000,
            dropout=mha_dropout)

        self.to_sequence_first = Transform(transform_fun=lambda x: x.permute(2, 0, 1), name='Permute(2, 0, 1)')

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=model_dim,
            dim_feedforward=model_dim * 2,
            nhead=mha_nhead,
            dropout=mha_dropout)

        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=mha_layers
        )

        self.to_batch_first = Transform(transform_fun=lambda x: x.permute(1, 2, 0), name='Permute(1, 2, 0)')

    def temporal_forward(self, x: Tensor) -> Tensor:
        x_enc = x * math.sqrt(self.model_dim)

        x_enc = self.to_sequence_first(x_enc)

        x_enc = self.pos_encoder(x_enc)

        src_mask = self.causal_mask(sz=len(x_enc), max_context= self.max_context)

        out = self.temporal_forward_loop(x=x_enc, src_mask=src_mask)

        out = self.to_batch_first(out)

        return out

    def temporal_forward_loop(self, x: Tensor, src_mask: Tensor) -> Tensor:
        if self.training:
            out = self.transformer_encoder(x, src_mask)
        else:
            seq_len = len(x)
        
        return out

    def causal_mask(
            self,
            sz: int,
            max_context: int | None = None) -> Tensor:
        """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        causal_mask  = torch.triu(
            torch.full((sz, sz), float('-inf')).to(device=self.device, dtype=self.dtype),
            diagonal=1,
        )

        if max_context is not None:
            max_context_mask = torch.tril(
                torch.full((sz, sz), float('-inf')).to(device=self.device, dtype=self.dtype),
                diagonal=-max_context,
            )
            causal_mask += max_context_mask

        return causal_mask

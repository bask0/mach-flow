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
from utils.torch_modules import DataTansform, EncodingModule, PadTau, Transform, SeqZeroPad, TemporalCombine


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


class TemporalNet(LightningNet, abc.ABC):

    def __init__(
            self,
            num_static_in: int,
            num_dynamic_in: int,
            num_outputs: int,
            model_dim: int = 8,
            enc_dropout: float = 0.0,
            fusion_method: str = 'post_repeated',
            **kwargs) -> None:

        super().__init__(**kwargs)

        self.fusion_method = fusion_method

        # Input encoding
        # -------------------------------------------------

        if self.fusion_method == 'pre_encoded':

            # Static input encoding
            self.static_encoding = EncodingModule(
                num_in=num_static_in,
                num_enc=model_dim,
                num_layers=2,
                dropout=enc_dropout,
                activation=torch.nn.ReLU(),
                activation_last=torch.nn.Tanh()
            )

            # Dynamic input encoding
            self.dynamic_encoding = EncodingModule(
                num_in=num_dynamic_in,
                num_enc=model_dim,
                num_layers=2,
                dropout=enc_dropout,
                activation=torch.nn.Tanh(),
                activation_last=torch.nn.Sigmoid()
            )

        elif self.fusion_method == 'pre_repeated':

            self.pre_fusion_layer = TemporalCombine(
                num_dynamic_in=num_dynamic_in,
                num_static_in=num_static_in,
                num_out=model_dim,
                num_layers=2,
                dropout=0.1,
            )

        elif self.fusion_method == 'post_repeated':

            self.input_layer = torch.nn.Conv1d(
                in_channels=num_dynamic_in,
                out_channels=model_dim,
                kernel_size=1
            )

        else:

            raise self.fusion_method_not_found(self.fusion_method)

        # Temporal layer
        # -------------------------------------------------

        # >>>> Is defined in subclass.

        # Output
        # -------------------------------------------------

        if self.fusion_method == 'post_repeated':
            self.post_fusion_layer = TemporalCombine(
                    num_dynamic_in=model_dim,
                    num_static_in=num_static_in,
                    num_out=model_dim,
                    num_layers=2,
                    dropout=0.1,
                )

        self.dropout1d = torch.nn.Dropout1d(enc_dropout)

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

    def temporal_pre_encoded(self, x: Tensor, s: Tensor | None, tau: float) -> Tensor:
        """Temoral layer with encoding pre-fusion.

        Args:
            x: The dynamic inputs, shape (batch, dynamic_channels, sequence)
            s: The static inputs, shape (batch, static_channels)
            tau: The optional tau probability for expectile/quantile regression.

        Returns:
            A tensor of predicted values (batch, outputs, sequence).

        """

        # Dynamic encoding: (B, D, S) -> (B, E, S)
        enc = self.dynamic_encoding(x)

        if s is not None:
            # Static encoding and unsqueezing: (B, C) ->  (B, E, 1)
            s_enc = self.static_encoding(s.unsqueeze(-1))

            # Multiply static encoding and dynamic encoding: (B, E, S) + (B, E, 1) -> (B, E, S)
            enc = enc + s_enc

        # Temporal layer and dropout: (B, E, S) -> (B, E, S)
        temp_enc = self.temporal_forward(enc)
        temp_enc = self.dropout1d(temp_enc)

        # Pad tau: (B, E, S) -> (B, E + 1, S)
        temp_enc = self.pad_tau(x=temp_enc, tau=tau)

        # Output layer, and activation: (B, E + 1, S) -> (B, O, S)
        out = self.output_layer(temp_enc)
        out = self.out_activation(out)

        return out

    def temporal_pre_repeated(self, x: Tensor, s: Tensor | None, tau: float) -> Tensor:
        """Temoral layer with repeated pre-fusion.

        Args:
            x: The dynamic inputs, shape (batch, dynamic_channels, sequence)
            s: The static inputs, shape (batch, static_channels)
            tau: The optional tau probability for expectile/quantile regression.

        Returns:
            A tensor of predicted values (batch, outputs, sequence).

        """

        # Fusion of dynamic and static features: (B, D, S), (B, C) -> (B, E, S)
        enc = self.pre_fusion_layer(x=x, s=s)

        # Temporal layer and dropout: (B, E, S) -> (B, E, S)
        temp_enc = self.temporal_forward(enc)
        temp_enc = self.dropout1d(temp_enc)

        # Pad tau: (B, E, S) -> (B, E + 1, S)
        temp_enc = self.pad_tau(x=temp_enc, tau=tau)

        # Output layer, and activation: (B, E + 1, S) -> (B, O, S)
        out = self.output_layer(temp_enc)
        out = self.out_activation(out)

        return out

    def temporal_post_repeated(self, x: Tensor, s: Tensor | None, tau: float) -> Tensor:
        """Temoral layer with repeated post-fusion.

        Args:
            x: The dynamic inputs, shape (batch, dynamic_channels, sequence)
            s: The static inputs, shape (batch, static_channels)
            tau: The optional tau probability for expectile/quantile regression.

        Returns:
            A tensor of predicted values (batch, outputs, sequence).

        """

        # Encoding of dynamic features: (B, D, S) -> (B, E, S)
        x_enc = self.input_layer(x)

        # Temporal layer: (B, E, S) -> (B, E, S)
        temp_enc = self.temporal_forward(x_enc)

        # Fusion of dynamic and static features and dropout: (B, E, S), (B, C) -> (B, E, S)
        temp_enc = self.post_fusion_layer(x=temp_enc, s=s)
        temp_enc = self.dropout1d(temp_enc)

        # Pad tau: (B, E, S) -> (B, E + 1, S)
        out = self.pad_tau(x=temp_enc, tau=tau)

        # Output layer, and activation: (B, E + 1, S) -> (B, O, S)
        out = self.output_layer(out)
        out = self.out_activation(out)

        return out

    def forward(self, x: Tensor, s: Tensor, tau: float) -> Tensor:

        if self.fusion_method == 'pre_encoded':
            return self.temporal_pre_encoded(x=x, s=s, tau=tau)
        elif self.fusion_method == 'pre_repeated':
            return self.temporal_pre_repeated(x=x, s=s, tau=tau)
        elif self.fusion_method == 'post_repeated':
            return self.temporal_post_repeated(x=x, s=s, tau=tau)
        else:
            raise self.fusion_method_not_found(self.fusion_method)

    def fusion_method_not_found(self, fusion_method: str) -> ValueError:
        return ValueError(
            f'`fusion_method` \'{fusion_method}\' is not defined.'
        )

    # def forward(self, x: Tensor, s: Tensor, tau: float) -> Tensor:

    #     if self.pre_fusion:

    #         # Fusion of dynamic and static features: (B, D, S), (B, C) -> (B, E, S)
    #         enc = self.pre_fusion_layer(x=x, s=s)

    #     else:

    #         # Encoding of dynamic features: (B, D, S) -> (B, E, S)
    #         enc = self.input_layer(x)

    #     # 2D dropout.
    #     enc = self.in_dropout1d(enc)

    #     # Temporal layer: (B, E, S) -> (B, E, S)
    #     out = self.temporal_forward(enc)

    #     if not self.pre_fusion:
    #         # Fusion of dynamic and static features: (B, E, S), (B, C) -> (B, E, S)
    #         out = self.post_fusion_layer(x=out, s=s)

    #     # 2D dropout.
    #     enc = self.out_dropout1d(enc)

    #     # Pad tau: (B, E, S) -> (B, E + 1, S)
    #     out = self.pad_tau(x=out, tau=tau)

    #     # Output layer, and activation: (B, E + 1, S) -> (B, O, S)
    #     out = self.output_layer(out)
    #     out = self.out_activation(out)

    #     return out

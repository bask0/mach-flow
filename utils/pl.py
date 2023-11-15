from typing import Any, Sequence
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary
import logging
from torch.utils.data import DataLoader
from torch import Tensor
import xarray as xr
import numpy as np
import os
import warnings

from utils.loss_functions import RegressionLoss
from utils.types import BatchPattern, ReturnPattern


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
            `x`, the sequencial input, and `m`, the meta-features.

    Shape:
        The subclass must implement a `forward` method (see PyTorch doc) which takes the arguments `x`, the
        sequencial input, and optional argument `s`, the static input:
        * `x`: (B, C, S)
        * `x`: (B, D)
        * return: (B, O, S)
        where B=batch size, S=sequence length, C=number of dynamic channels, and D=number of static channels.
    """
    def __init__(self, **kwargs) -> None:
        """Initialize lightning module, should be subclassed.

        Args:
            kwargs:
                Do not use kwargs, required as sink for exceeding arguments due to pytorch ligthning's argparse scheme.
        """

        super().__init__()

        self.loss_fn = RegressionLoss(criterion='l1', sample_wise=False)

    def shared_step(
            self,
            batch: BatchPattern,
            step_type: str,
            batch_idx: int) -> tuple[Tensor, ReturnPattern]:
        """A single training step shared across specialized steps that returns the loss and the predictions.

        Args:
            batch: the bach.
            step_type: the step type (training mode), one of (`train`, `val`, `test`, `pred`).

        Returns:
            tuple[Tensor, ReturnPattern]: the loss, the predictions.
        """

        if step_type not in ('train', 'val', 'test', 'pred'):
            raise ValueError(f'`step_type` must be one of (`train`, `val`, `test`, `pred`), is {step_type}.')

        target_hat = self(
            x=batch.dfeatures,
            s=batch.sfeatures
        )

        num_cut = batch.coords.warmup_size[0]
        batch_size = target_hat.shape[0]

        loss = self.loss_fn(
            input=target_hat[..., num_cut:],
            target=batch.dtargets[..., num_cut:]
        )

        preds = ReturnPattern(
            dtargets=target_hat,
            coords=batch.coords
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

        loss, _ = self.shared_step(batch, step_type='train', batch_idx=batch_idx)

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

        loss, _ = self.shared_step(batch, step_type='val', batch_idx=batch_idx)

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

        loss, _ = self.shared_step(batch, step_type='test', batch_idx=batch_idx)

        return {'test_loss': loss}

    def predict_step(
            self,
            batch: BatchPattern,
            batch_idx: int,
            dataloader_idx: int = 0
            ) -> ReturnPattern:
        """A single predict step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        """

        _, preds = self.shared_step(batch, step_type='pred', batch_idx=batch_idx)

        return preds

    def summarize(self):
        s = f'=== Summary {"=" * 31}\n'
        s += f'{str(ModelSummary(self))}\n\n'
        s += f'=== Model {"=" * 33}\n'
        s += f'{str(self)}'

        return s


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str):
        super().__init__(write_interval="epoch")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_on_epoch_end(
            self,
            trainer: 'pl.Trainer',
            pl_module: 'pl.LightningModule',
            predictions: Sequence[ReturnPattern],
            batch_indices: Sequence[Any]) -> None:

        pdl: DataLoader = trainer.predict_dataloaders

        warmup_size: int = pdl.dataset.warmup_size
        ds: xr.Dataset = pdl.dataset.ds
        out_ds = xr.Dataset()

        encoding = {}
        for t, target in enumerate(pdl.dataset.targets):
            new_target_name = target + '_mod'
            da = xr.full_like(ds[target], np.nan).compute()
            for output in predictions:
                coords = output.coords
                for i, (station, start_date, end_date) in enumerate(zip(coords.station, coords.start_date, coords.end_date)):
                    da.loc[{'station': station, 'time': slice(start_date, end_date)}] = \
                        output.dtargets[i, t, warmup_size:].detach().numpy()

            out_ds[new_target_name] = da

            encoding.update(
                {
                    new_target_name: {
                        'chunks': (1, -1)
                    }
                }
            )

        write_path = self.make_predition_path(self.output_dir)

        out_ds.to_zarr(store=write_path, mode='w', encoding=encoding)

        logger.info(f'Predictions written to \'{write_path}\'.')

    @staticmethod
    def make_predition_path(output_dir: str) -> str:
        return os.path.join(output_dir, 'preds.zarr')

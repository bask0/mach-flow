from typing import Any, Sequence
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import BasePredictionWriter
import logging
from torch.utils.data import DataLoader
import xarray as xr
import numpy as np
import os
import sys
import optuna
import warnings
from typing import TYPE_CHECKING

from utils.types import ReturnPattern

if TYPE_CHECKING:
    from utils.tuning import SearchSpace

# Ignore anticipated PL warnings.
warnings.filterwarnings('ignore', '.*infer the indices fetched for your dataloader.*')
warnings.filterwarnings('ignore', '.*You requested to overfit.*')

logger = logging.getLogger('lightning')


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str):
        super().__init__(write_interval="epoch")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_on_epoch_end(
            self,
            trainer: 'pl.Trainer',
            pl_module: 'pl.LightningModule',
            predictions: Sequence[list[ReturnPattern]],
            batch_indices: Sequence[Any]) -> None:

        pdl: DataLoader = trainer.predict_dataloaders

        warmup_size: int = pdl.dataset.warmup_size
        ds: xr.Dataset = pdl.dataset.ds
        out_ds = ds.copy()

        taus = [p.tau for p in predictions[0]]

        encoding = {}

        for var in ds.data_vars:
            if 'time' in ds[var].dims:
                ck = (40, -1)
            else:
                ck = (40,)
            encoding.update(
                {
                    var: {
                        'chunks': ck,
                    }
                }
            )

        for t, target in enumerate(pdl.dataset.targets):
            new_target_name = target + '_mod'
            da = xr.full_like(ds[target], np.nan).expand_dims(tau=taus).copy().compute()
            for outputs in predictions:
                for output in outputs:
                    coords = output.coords
                    tau = output.tau
                    for i, (station, start_date, end_date) in enumerate(
                            zip(coords.station, coords.start_date, coords.end_date)):
                        da.loc[{
                            'station': station,
                            'time': slice(start_date, end_date),
                            'tau': tau
                        }] = output.dtargets[i, t, warmup_size:].detach().numpy()

            out_ds[new_target_name] = da

            encoding.update(
                {
                    new_target_name: {
                        'chunks': (1, 40, -1),
                    }
                }
            )

        write_path = self.make_predition_path(self.output_dir)

        out_ds.to_zarr(store=write_path, mode='w', encoding=encoding)

        logger.info(f'Predictions written to \'{write_path}\'.')

    @staticmethod
    def make_predition_path(output_dir: str) -> str:
        return os.path.join(output_dir, 'preds.zarr')



class MyLightningCLI(LightningCLI):
    def __init__(
            self,
            directory: str,
            version: str | int,
            version_prefix: str = '',
            add_prediction_writer_callback: bool = False,
            *args,
            **kwargs):

        self.directory = directory

        if isinstance(version, str):
            self.version = version
        else:
            self.version = self.id2version(prefix=version_prefix, id=version)

        if 'parser_kwargs' not in kwargs:
            kwargs['parser_kwargs'] = {}

        kwargs['parser_kwargs'].update({'parser_mode': 'omegaconf'})

        if 'parser_kwargs' not in kwargs:
            kwargs['parser_kwargs'] = {}

        if add_prediction_writer_callback:
            kwargs.setdefault('trainer_defaults', {})
            kwargs['trainer_defaults'].setdefault('callbacks', [])

            if not isinstance(kwargs['trainer_defaults']['callbacks'], list):
                raise ValueError(
                    '\'trainer_defaults[\'callbacks\']\' must be list if passed.'
                )

            kwargs['trainer_defaults']['callbacks'].append(PredictionWriter(output_dir=self.version_dir))

        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        # Training routine args
        # -------------------------------------
        parser.add_argument(
            '--dev',
            action='store_true',
            help='quick dev run with one epoch and 1 batch.')
        parser.add_argument(
            '--skip_tuning',
            action='store_true',
            help='skip tuning and do xval; tuning must be present.')

        # Criterion routine args
        # -------------------------------------
        parser.add_argument(
            '--criterion.name',
            type=str,
            default='l2',
            help='the criterion to use for optimization, defauls to \'l2\'.')
        parser.add_argument(
            '--criterion.sqrt_transform',
            action='store_true',
            help='whether to compute loss on sqrt transformed scale.')
        parser.add_argument(
            '--criterion.inference_taus',
            nargs='+',
            type=float,
            default=[0.05, 0.25, 0.5, 0.75, 0.95],
            help='tau values to evaluate in inference; only applies for distribution-aware criterions.')

        # Linking args
        # -------------------------------------
        parser.link_arguments(
            'data.num_sfeatures', 'model.init_args.num_static_in', apply_on='instantiate')
        parser.link_arguments(
            'data.num_dfeatures', 'model.init_args.num_dynamic_in', apply_on='instantiate')
        parser.link_arguments(
            'data.num_dtargets', 'model.init_args.num_outputs', apply_on='instantiate')
        parser.link_arguments(
            'data.norm_args_features', 'model.init_args.norm_args_features', apply_on='instantiate')
        parser.link_arguments(
            'data.norm_args_stat_features', 'model.init_args.norm_args_stat_features', apply_on='instantiate')
        parser.link_arguments(
            'data.norm_args_targets', 'model.init_args.norm_args_targets', apply_on='instantiate')
        parser.link_arguments(
            'criterion.name', 'model.init_args.criterion', apply_on='parse')
        parser.link_arguments(
            'criterion.sqrt_transform', 'model.init_args.sqrt_transform', apply_on='parse')
        parser.link_arguments(
            'criterion.inference_taus', 'model.init_args.inference_taus', apply_on='parse')

    @staticmethod
    def id2version(prefix: str, id: int) -> str:
        return f'{prefix}_{id:03d}'

    @property
    def subc(self) -> str:
        if self.subcommand is None:
            return ''
        else:
            return f'{self.subcommand}.'

    @property
    def version_dir(self) -> str:
        return os.path.join(self.directory, self.version)

    @property
    def best_checkpoint_dir(self) -> str:
        return os.path.join(self.version_dir, 'checkpoints', 'best.ckpt')

    @property
    def config_dir(self) -> str:
        return os.path.join(self.version_dir, 'config.yaml')

    def before_instantiate_classes(self):

        if self.config['dev']:
            self.config['trainer']['limit_train_batches'] = 2
            self.config['trainer']['limit_val_batches'] = 2
            self.config['trainer']['max_epochs'] = 2

        self.config[f'{self.subc}trainer.logger'] = {
            'class_path': 'lightning.pytorch.loggers.tensorboard.TensorBoardLogger',
            'init_args': {
                'save_dir': self.directory,
                'version': self.version,
                'name': ''
            }
        }

    # @staticmethod
    # def link_optimizers_and_lr_schedulers(parser):
    #     # Set lr_scheduler's num_training_steps from datamodule class
    #     parser.link_arguments(
    #         'optimizer.init_args.lr', 'lr_scheduler.init_args.max_lr', apply_on='parse')
    #     # parser.link_arguments(
    #     #     'data.num_steps_per_epoch', 'lr_scheduler.init_args.steps_per_epoch', apply_on='instantiate')
    #     # parser.link_arguments(
    #     #     'trainer.max_epochs', 'lr_scheduler.init_args.epochs', apply_on='parse')
    #     parser.link_arguments(
    #         'trainer.max_epochs', 'lr_scheduler.init_args.total_steps', apply_on='parse')

    #     LightningCLI.link_optimizers_and_lr_schedulers(parser)


def get_dummy_cli() -> MyLightningCLI:

    config_file = os.path.join(os.path.dirname(sys.argv[0]), 'config.yaml')
    
    cli = MyLightningCLI(
        directory='',
        version='',
        run=False,
        parser_kwargs={
            'default_config_files': [config_file]
        })

    return cli


def cli_main(
        trial: optuna.Trial,
        directory: str,
        is_tune: bool,
        config_paths: list[str] | str | None,
        search_space: 'SearchSpace | None' = None,
        **kwargs) -> float:

    if config_paths is None:
        config_paths = []
    elif isinstance(config_paths, str):
        config_paths = [config_paths]

    if search_space is None:
        raise ValueError('with \'is_predict=False\', search_space must be passed.')

    config_paths.append(search_space.config_path)

    cli = MyLightningCLI(
        directory=directory,
        version=trial._trial_id if is_tune else trial.params['fold_nr'],
        version_prefix='trial' if is_tune else 'fold',
        add_prediction_writer_callback=not is_tune,
        run=False,
        parser_kwargs={
            'default_config_files': config_paths
        },
        **kwargs
    )

    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)

    trial.set_user_attr(
        'best_checkpoint', cli.best_checkpoint_dir)
    trial.set_user_attr(
        'config', cli.config_dir)
    trial.set_user_attr(
        'epoch', cli.trainer.current_epoch)

    # val_loss = cli.trainer.callback_metrics['val_loss'].item()
    val_loss = cli.trainer.early_stopping_callback.best_score.item()

    if not is_tune:
        model = type(cli.model).load_from_checkpoint(cli.best_checkpoint_dir)
        cli.trainer.predict(model=model, datamodule=cli.datamodule)

    return val_loss

import os
import sys
import shutil
import optuna

from optuna.integration import PyTorchLightningPruningCallback

from project.machflowdata import MachFlowDataModule
from project.lstm_regressor import LSTM
from utils.pl import MyLightningCLI, PredictionWriter
from utils.tuning import SearchSpace, FakeTrial

LOG_DIR = './logs'
DEFAULT_CONFIG_PATH = 'config/default_config.yaml'



class LSTMSearchSpace(SearchSpace):
    """Defines the search space for the LSTM model."""
    def __init__(self, trial: optuna.Trial) -> None:
        config = {
            'model': {
                'num_enc': trial.suggest_int('num_enc', 2, 10),
            },
            'optimizer': {
                'class_path': 'torch.optim.AdamW',
                'init_args': {
                    'lr': trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2]),
                    'weight_decay': trial.suggest_categorical('weight_decay', [0, 1e-4, 1e-2]),
                }
            }
        }
        super().__init__(config=config)



def cli_tune(trial: optuna.Trial | FakeTrial, exp_name: str) -> float:

    dryrun = isinstance(trial, FakeTrial)

    trial_dir = os.path.join(LOG_DIR, exp_name, MyLightningCLI.trial2version(trial))

    if dryrun:
        config_files = [DEFAULT_CONFIG_PATH]
        trainer_callbacks = []
        search_space = None
    else:
        search_space = LSTMSearchSpace(trial=trial)
        config_files = [DEFAULT_CONFIG_PATH, search_space.config_path]
        trainer_callbacks = [
            PyTorchLightningPruningCallback(
                trial=trial,
                monitor='val_loss'),
            PredictionWriter(
                output_dir=trial_dir
            )
        ]

    try:
        cli = MyLightningCLI(
            log_dir=LOG_DIR,
            version=MyLightningCLI.trial2version(trial),
            exp_name=exp_name,
            model_class=LSTM,
            datamodule_class=MachFlowDataModule,
            parser_kwargs=MyLightningCLI.expand_to_subcommands({'default_config_files': config_files}),
            trainer_defaults={'callbacks': trainer_callbacks},
        )

    finally:
        if search_space is not None:
            search_space.teardown()

    if dry_run:
        return 0

    val_loss = cli.trainer.callback_metrics['val_loss'].item()

    config_path = os.path.join(trial_dir, 'config.yaml')
    prediction_path = PredictionWriter.make_predition_path(trial_dir)

    trial.set_user_attr(
        'config_path', config_path)
    trial.set_user_attr(
        'prediction_path', prediction_path)
    trial.set_user_attr(
        'epoch', cli.trainer.current_epoch)

    cli.trainer.predict(datamodule=cli.datamodule, ckpt_path='best')

    return val_loss


if __name__ == '__main__':

    exp_name = 'test'
    exp_dir = os.path.join(LOG_DIR, exp_name)
    db_path = os.path.join(exp_dir, 'optuna.db')

    subcommand = sys.argv[1] if len(sys.argv) > 1 else ''

    dry_run = any([arg in sys.argv for arg in ['-h', '--help', '--print_config']])

    # TUNING
    # =================
    if subcommand == 'fit':
        if dry_run:
            cli_tune(trial=FakeTrial(), exp_name=exp_name)

        if os.path.exists(exp_dir) and not dry_run:
            shutil.rmtree(exp_dir)

        os.makedirs(exp_dir, exist_ok=True)

        pruner = optuna.pruners.MedianPruner(n_startup_trials=4) if True else optuna.pruners.NopPruner()

        study = optuna.create_study(
            storage=f'sqlite:///{db_path}',
            study_name=exp_name,
            direction='minimize',
            pruner=pruner,
            load_if_exists=dry_run
        )
        study.optimize(lambda trial: cli_tune(trial=trial, exp_name=exp_name), n_trials=20, timeout=600)

    # INVALID OPTIONS
    # =================
    else:
        raise ValueError(
            f'only the subcommands \'fit\' and \'predict\' are valid, \'{subcommand}\' was passed.'
        )

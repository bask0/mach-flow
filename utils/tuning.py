import os
import sys
import shutil
import yaml
import tempfile
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from typing import Type, Callable, TypeVar, TYPE_CHECKING

from utils.pl import cli_main, get_dummy_cli, LightningNet
from utils.analysis import study_summary, plot_xval_cdf

if TYPE_CHECKING:
    from utils.pl import MyLightningCLI

import logging
logger = logging.getLogger('pytorch_lightning')


class SearchSpaceMeta(type): 
    def __new__(cls, name: str, bases, defcl) -> 'SearchSpace':
        obj: 'SearchSpace' = super().__new__(cls, name, bases, defcl)

        if obj.MODEL is None:
            obj.MODEL_PATH = None
            obj.MODEL_NAME = None
        elif isinstance(obj.MODEL, type) and issubclass(obj.MODEL, LightningNet):
            path, name = cls.classname(obj.MODEL)
            obj.MODEL_PATH = path
            obj.MODEL_NAME = name
        else:
            raise TypeError(
                f'class attribute `MODEL` of {type(obj).__name__} must be `None` or a subclass of `LightningNet`, '
                f'is `{type(obj.MODEL).__name__}`.'
            )

        return obj

    @classmethod
    def classname(cls, model_class: 'Type[LightningNet]') -> tuple[str, str]:
        module = model_class.__module__
        name = model_class.__qualname__
        if module is not None and module != "__builtin__":
            path = module + "." + name
        else:
            path = name
        return path, name


class SearchSpace(object, metaclass=SearchSpaceMeta):
    """SearchSpace definition, meant to be subclassed.

    This base SearchSpace must be subclassed.
    * In the subclasse\'s `CustomSearchSpace.__init__` method, the parent class muts be initialized
    using `super().__init__(trial=..., config=...)`.
    * The subclass must override the class attribute `MODEL` with the `utils.pl.LightningNet` for which the search space configuration applies.
    * The subclasse\'s `CustomSearchSpace.__init__` method must take a single argument, `trial`, an `optuna.Trial`.
    * The config file is a yaml-style configuration which is passed to the LightningCLI.
    * The search space is constructed using optune trials.
    * The default optimizer (Ada) can be changed by overriding the class attribute `OPTIMIZER_PATH` in the subclass.

    Example:
    >>> import MyLightningModel
        class CustomSearchSpace(SearchSpace):
            MODEL = MyLightningModel  # Must be subclass of utils.pl.LightningNet
            OPTIMIZER_PATH = 'torch.optim.SGD'  # use SGD instead of AdamW.
            def __init__(self, trial: optuna.Trial) -> None:
                config = {
                    'model': {
                        'num_layers': trial.suggest_int('num_layers', 1, 3),
                    },
                    'optimizer': {
                        'lr': trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2]),
                        'weight_decay': trial.suggest_categorical('weight_decay', [0, 1e-4, 1e-2]),
                }
                super().__init__(trial=trial, config=config)

    """

    MODEL: Type[LightningNet] | None = None
    MODEL_PATH: str | None
    MODEL_NAME: str | None
    OPTIMIZER_PATH: str = 'torch.optim.AdamW'

    IS_XVAL: bool = False

    def __init__(self, trial: optuna.Trial, config: dict | None = None) -> None:
        """Initialize the class. Muts be called at the end of subclass initialization.

        Args:
            trial (optuna.Trial): An optuna trial.
            config (dict): the configuration, see documentation for more details. the argument is required, the       
                default (None) is used to suppress subclass type hints.
        """

        if config is None:
            raise ValueError(
                'you must provide \'config\' argument other than None.'
            )

        self._validate_model_path()

        self.trial = trial
        self.config = self.make_config(config)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.config_path = f.name

        self._dump_config()

    def _validate_model_path(self) -> None:
        if self.IS_XVAL:
            return

        if self.MODEL_PATH is None:
            raise ValueError(
                'you must override the attribute `MODEL_PATH` in the subclass.'
            )

    def make_config(self, config: dict) -> dict:
        new_config = {}

        for key, value in config.items():
            if key == 'model':
                model_config = {
                    'model': {
                        'class_path': self.MODEL_PATH,
                        'init_args': value
                    }
                }
                new_config.update(**model_config)
            elif key == 'optimizer':
                optimizer_config = {
                    'optimizer': {
                        'class_path': self.OPTIMIZER_PATH,
                        'init_args': value
                    }
                }
                new_config.update(**optimizer_config)
            else:
                new_config.update(**{key: value})

        return new_config

    @property
    def config_path(self) -> str:
        if not hasattr(self, '_config_path'):
            raise RuntimeError(
                f'cannot get \'config_path\' from \'{type(self).__name__}\'. '
                'Most likely, the parent class initialization (`super().__init__`) '
                'has not been called in the subclass initialization.'
            )

        return self._config_path

    @config_path.setter
    def config_path(self, value: str) -> None:
        self._config_path = value

    def _dump_config(self):
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file)

    def teardown(self) -> None:
        os.remove(self.config_path)


def get_cv_search_space(num_folds: int) -> Type[SearchSpace]:
    class CVSearchSpace(SearchSpace):
        """Defines the cross validation search space for iteration across folds."""

        IS_XVAL = True

        def __init__(self, trial: optuna.Trial) -> None:
            config = {
                'data': {
                    'class_path': 'dataset.machflowdata.MachFlowDataModule',
                    'init_args': {
                        'fold_nr': trial.suggest_categorical('fold_nr', list(range(num_folds)))
                    }
                },
            }
            super().__init__(trial=trial, config=config)

    return CVSearchSpace


def get_model_name_from_cli(cli: 'MyLightningCLI') -> str:
    return type(cli.model).__name__


def get_default_config_file() -> str:
    config_file =  os.path.join(os.path.dirname(sys.argv[0]), 'config.yaml')
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            'default config file not found. a config file named `config.yaml` must be present '
            f'in the calling script directory at:\n  > {config_file}'
        )
    return config_file


def get_exp_name() -> str:
    exp_name =  os.path.basename(os.path.abspath(os.path.dirname(sys.argv[0])))
    return exp_name


def get_config_name(cli) -> str:

    name = ''

    if cli.config['config'] is not None:
        for path in cli.config['config']:
            rel_path = path.rel_path
            file_name = os.path.basename(rel_path)
            if not file_name.endswith('.yaml'):
                raise ValueError(
                    f'config files must have ending \'.yaml\', found {rel_path}.'
                )
            name += '_' + file_name.split('.yaml')[0]

        name = name.strip('_')

    if name == '':
        name = 'default'

    return name


def get_search_spaces():
    d = {cls.MODEL_NAME: cls for cls in SearchSpace.__subclasses__()}
    if len(d) < 1:
        raise RuntimeError(
            'no SearchSpace detected.'
        )
    return d


SS = TypeVar('SS', bound='SearchSpace')

class Tuner(object):
    def __init__(
            self,
            sampler: optuna.samplers.BaseSampler,
            pruner: optuna.pruners.BasePruner,
            log_dir: str = 'runs',
            overwrite: bool = True) -> None:

        cli = get_dummy_cli()

        self.is_dev = cli.config['dev']
        self.skip_tuning = cli.config['skip_tuning']
        search_spaces = get_search_spaces()

        self.sampler = sampler
        self.pruner = pruner
        self.log_dir = log_dir
        self.exp_name = get_exp_name()
        self.config_name = get_config_name(cli)
        self.model = get_model_name_from_cli(cli)
        self.num_xval_folds = cli.config['data']['init_args']['num_cv_folds']
        self.overwrite = overwrite

        if self.model not in search_spaces:
            raise KeyError(
                f'for tuning of the model class {self.model}, a `search_space` with the '
                f'key \'{self.model}\' must be provided.'
            )

        self.search_space = search_spaces[self.model]

        self.exp_path_tune = os.path.join(self.log_dir, self.exp_name, self.config_name, self.model, 'tune')
        self.db_path_tune = os.path.join(self.exp_path_tune, 'optuna.db')

        self.exp_path_xval = os.path.join(self.log_dir, self.exp_name, self.config_name, self.model, 'xval')
        self.db_path_xval = os.path.join(self.exp_path_xval, 'optuna.db')

        self.test_time_slice = self.get_test_slice(cli)
        self.best_config_path = None

    def new_study(self, db_path: str, is_tune: bool) -> optuna.Study:
        study = optuna.create_study(
            storage=f'sqlite:///{db_path}',
            study_name=self.model,
            direction='minimize',
            sampler=self.sampler if is_tune else optuna.samplers.BruteForceSampler(),
            pruner=self.pruner if is_tune else optuna.pruners.NopPruner(),
            load_if_exists=False
        )

        return study

    @property
    def tune_study(self) -> optuna.Study:
        if not os.path.exists(self.db_path_tune):
            raise FileNotFoundError(
                f'tune study does not exist: {self.db_path_tune}. Did you `RunConfig.tune` already?'
            )
        return optuna.load_study(study_name=self.model, storage=f'sqlite:///{self.db_path_tune}')

    @property
    def xval_study(self) -> optuna.Study:
        if not os.path.exists(self.db_path_xval):
            raise FileNotFoundError(
                f'xval study does not exist: {self.db_path_tune}. Did you `RunConfig.xval` already?'
            )
        return optuna.load_study(study_name=self.model, storage=f'sqlite:///{self.db_path_xval}')

    def get_objective(self, is_tune: bool) -> Callable[[optuna.Trial], float]:
        if is_tune:
            search_space = self.search_space
        else:
            search_space = get_cv_search_space(num_folds=self.num_xval_folds)

        def objective(trial) -> float:
            if is_tune:
                config_file = get_default_config_file()

                kwargs = {
                    'trainer_defaults':
                        {'callbacks': [
                            PyTorchLightningPruningCallback(trial=trial, monitor='val_loss')
                        ]}
                }
            else:
                config_file, _ = self.get_best_config_and_ckpt(self.tune_study)

                kwargs = {}

            return cli_main(
                trial=trial,
                directory=self.exp_path_tune if is_tune else self.exp_path_xval,
                is_tune=is_tune,
                config_paths=config_file,
                search_space=search_space(trial),
                **kwargs
            )

        return objective

    def tune(self, n_trials: int, **kwargs) -> None:

        if self.skip_tuning:
            logger.warning('Tuning is skipped ue to \'--skip_tuning\'.')
            if not os.path.exists(self.exp_path_tune):
                raise RuntimeError(
                    'no tuning path found but \'--skip_tuning\'. Did you run tuning beforehand?'
                )
            return

        if self.is_dev:
            logger.warning('Setting `n_trials=2` for dev run.')
            n_trials = 2

        if os.path.exists(self.exp_path_tune) and self.overwrite:
            shutil.rmtree(self.exp_path_tune)

        os.makedirs(self.exp_path_tune)

        study = self.new_study(db_path=self.db_path_tune, is_tune=True)

        study.optimize(self.get_objective(is_tune=True), n_trials=n_trials, **kwargs)
        self.summarize_tuning()

    def xval(self) -> None:
        if os.path.exists(self.exp_path_xval) and self.overwrite:
            shutil.rmtree(self.exp_path_xval)

        os.makedirs(self.exp_path_xval)

        study = self.new_study(db_path=self.db_path_xval, is_tune=False)
        study.optimize(self.get_objective(is_tune=False))
        self.plot_xval_cdf()

    @staticmethod
    def get_best_config_and_ckpt(study: optuna.Study) -> tuple[str, str]:
        best_trial = study.best_trial

        return best_trial.user_attrs['config'], best_trial.user_attrs['best_checkpoint']

    @staticmethod
    def get_all_config_and_ckpt(study: optuna.Study) -> list[tuple[optuna.Trial, str, str]]:
        configs_and_ckpts = []

        for trial in study.trials:
            configs_and_ckpts.append(
                (   trial,
                    trial.user_attrs['config'],
                    trial.user_attrs['best_checkpoint']
                )
            )

        return configs_and_ckpts

    def summarize_tuning(self) -> None:
        study_summary(study_path=self.db_path_tune, study_name=self.model)

    @staticmethod
    def get_test_slice(cli: 'MyLightningCLI') -> slice:
        try:
            test_range = cli.config['data']['init_args']['test_date_slice']
        except KeyError as e:
            raise KeyError(
                'could not infer test set time range from config at `data.init_args.test_date_slice`.'
            )

        return slice(*test_range)    

    def plot_xval_cdf(self) -> None:
        plot_xval_cdf(
            xval_dir=self.exp_path_xval,
            save_path=os.path.join(self.exp_path_xval, 'xval_perf.png'),
            subset={'time': self.test_time_slice}
        )

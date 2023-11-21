import os
import shutil
import sys
import yaml
import tempfile
import optuna
from typing import Type, Callable

from utils.pl import cli_main
from utils.analysis import study_summary, plot_xval_cdf


class SearchSpace(object):
    """SearchSpace definition, meant to be subclassed.

    This base SearchSpace must be subclassed.
    * In the subclasse\'s `CustomSearchSpace.__init__` method, the parent class muts be initialized
    using `super().__init__(config=...)`.
    * The subclasse\'s `CustomSearchSpace.__init__` method must take a single argument, `trial`, an `optuna.Trial`.
    * The config file is a yaml-style configuration which is passed to the LightningCLI.
    * The search space is constructed using optune trials.

    Example:
    >>> class CustomSearchSpace(SearchSpace):
        def __init__(self, trial: optuna.Trial) -> None:
            config = {
                'model': {
                    'num_layers': trial.suggest_int('num_layers', 1, 3),
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

    """
    def __init__(self, trial: optuna.Trial, config: dict) -> None:
        """Initialize the class. Muts be called at the end of subclass initialization.

        Args:
            config (dict): the configuration, see documentation for more details.
        """

        self.trial = trial
        self.config = config

        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.config_path = f.name

        self._dump_config()

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


class RunConfig(object):
    def __init__(self, log_dir: str, exp_name: str, is_tune: bool, overwrite: bool = True) -> None:
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.model = self.infer_model()
        self.is_tune = is_tune
        self.subdir = 'tune' if is_tune else 'xval'

        self.exp_path = os.path.join(self.log_dir, self.exp_name, self.model, self.subdir)
        self.db_path = os.path.join(self.exp_path, 'optuna.db')

        if os.path.exists(self.exp_path) and overwrite:
            shutil.rmtree(self.exp_path)

        os.makedirs(self.exp_path)

        self.search_spaces = {}

    def infer_model(self) -> str:
        model = '<not found>'
        for i in range(len(sys.argv)):
            if sys.argv[i] == '-m':
                if (i + 1) >= len(sys.argv):
                    raise RuntimeError(
                        'no `-m` argument passed?'
                    )
                model = sys.argv[i + 1].lower()

        if model == '<not found>':
            raise RuntimeError(
                'parameter `-m` missing.'
            )

        return model

    def register_search_spaces(self, **search_space_class: Type[SearchSpace]) -> None:
        self.search_spaces.update(**search_space_class)

    def get_search_space(self, trial: optuna.Trial) -> SearchSpace:
        if self.is_tune:
            if (self.model not in self.search_spaces):
                raise KeyError(
                    f'no SearchSpace with name \'{self.model}\' has been registered for HP tuning. '
                    'Do so with `this.register_search_spaces`.'
                )
            expected_search_space_name = self.model
        else:
            if 'xval' not in self.search_spaces:
                raise KeyError(
                    'no SeaerchSpace with name \'xval\' has been registered. This is required to run '
                    'cross validation.'
                )
            expected_search_space_name = 'xval'

        return self.search_spaces[expected_search_space_name](trial)

    def get_study(
            self,
            sampler: optuna.samplers.BaseSampler = optuna.samplers.RandomSampler(),
            pruner: optuna.pruners.BasePruner = optuna.pruners.NopPruner()) -> optuna.Study:

        study = optuna.create_study(
            storage=f'sqlite:///{self.db_path}',
            study_name=self.model,
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            load_if_exists=False
        )

        return study

    def get_objective(self, config_path: str | None = None) -> Callable[[optuna.Trial], float]:
        def objective(trial) -> float:
            return cli_main(
                trial=trial,
                directory=self.exp_path,
                exp_name=self.exp_name,
                is_tune=self.is_tune,
                config_path=config_path,
                search_space=self.get_search_space(trial=trial))

        return objective

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

    def predict_trials(self, study: optuna.Study) -> None:
        for trial, config, ckpt in self.get_all_config_and_ckpt(study):

            print('>>> Prediction with best model:')
            print(f'    - config: {config}')
            print(f'    - ckpt:   {ckpt}')

            cli_main(
                trial=trial,
                directory=self.exp_path,
                exp_name=self.exp_name,
                is_tune=False,
                is_predict=True,
                search_space=None,
                config_path=config,
                ckpt_path=ckpt,
                save_config_callback=None
            )

        if not self.is_tune:
            self.plot_xval_cdf()

    def summarize_tuning(self) -> None:
        study_summary(study_path=self.db_path, study_name=self.model)

    def plot_xval_cdf(self) -> None:
        plot_xval_cdf(xval_dir=self.exp_path, save_path=os.path.join(self.exp_path, 'xval_perf.png'))

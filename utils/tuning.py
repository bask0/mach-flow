import os
import shutil
import sys
import yaml
import tempfile
import optuna
from typing import Type, Callable

from utils.pl import cli_main
from utils.types import PredictTrial
from utils.analysis import study_summary


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


class TuneConfig(object):
    def __init__(self, log_dir: str, exp_name: str, overwrite: bool = True) -> None:
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.model = self.infer_model()

        self.exp_path = os.path.join(self.log_dir, self.exp_name, self.model)
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
        if self.model not in self.search_spaces:
            raise KeyboardInterrupt(
                f'no SearchSpace with name {self.model} has been registered. do so with `this.register_search_spaces`.'
            )

        return self.search_spaces[self.model](trial)

    def get_study(self, pruner: optuna.pruners.BasePruner = optuna.pruners.NopPruner()) -> optuna.Study:
        study = optuna.create_study(
            storage=f'sqlite:///{self.db_path}',
            study_name=self.model,
            direction='minimize',
            pruner=pruner,
            load_if_exists=False
        )
        return study

    def get_objective(self) -> Callable[[optuna.Trial], float]:
        def objective(trial) -> float:
            return cli_main(
                trial=trial,
                directory=self.exp_path,
                exp_name=self.exp_name,
                search_space=self.get_search_space(trial=trial))

        return objective

    def predict_with_best_model(self, study: optuna.Study) -> None:
        best_trial = study.best_trial

        config = best_trial.user_attrs['config']
        ckpt = best_trial.user_attrs['best_checkpoint']

        print('>>> Prediction with best model:')
        print(f'    - config: {config}')
        print(f'    - ckpt:   {ckpt}')

        cli_main(
            trial=PredictTrial(
                config_dir=config,
                best_model_path=ckpt
            ),
            directory=self.exp_path,
            exp_name=self.exp_name,
            search_space=None
        )

    def summarize_tuning(self) -> None:
        study_summary(study_path=self.db_path, study_name=self.model)

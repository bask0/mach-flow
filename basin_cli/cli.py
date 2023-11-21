import optuna

from models.lstm import LSTM  # noqa: F401
from utils.tuning import SearchSpace, Tuner


class LSTMSearchSpace(SearchSpace):
    """Defines the search space for the LSTM model."""

    MODEL_PATH = 'models.lstm.LSTM'
    
    def __init__(self, trial: optuna.Trial) -> None:
        config = {
            'model': {
                'num_enc': trial.suggest_int('num_enc', 2, 10, step=2)
            },
            'optimizer': {
                'lr': trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2]),
                'weight_decay': trial.suggest_categorical('weight_decay', [0, 1e-4, 1e-2]),
            }
        }
        super().__init__(trial=trial, config=config)


SEARCH_SPACES = {
    'LSTM': LSTMSearchSpace
}


if __name__ == '__main__':

    # HP tuning
    # ----------------------
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4)
    sampler = optuna.samplers.RandomSampler()
    tuner = Tuner(
        sampler=sampler,
        pruner=pruner,
        search_spaces=SEARCH_SPACES,
        )

    tuner.tune(n_trials=20)
    tuner.xval()

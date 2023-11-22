import optuna

from models.lstm import LSTM
from utils.tuning import SearchSpace, Tuner


class LSTMSearchSpace(SearchSpace):
    """Defines the search space for the LSTM model."""

    MODEL = LSTM

    def __init__(self, trial: optuna.Trial) -> None:
        config = {
            'model': dict(
                num_hidden=trial.suggest_categorical('num_hidden', [16, 32, 64]),
                num_enc=trial.suggest_categorical('num_enc', [4, 8, 16]),
                enc_dropout=trial.suggest_categorical('enc_dropout', [0.0, 0.2]),
                num_enc_layers=trial.suggest_categorical('num_enc_layers', [1, 2]),
                num_lstm_layers=trial.suggest_categorical('num_lstm_layers', [1, 2]),
            ),
            'optimizer': dict(
                lr=trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2]),
                weight_decay=trial.suggest_categorical('weight_decay', [0, 1e-4, 1e-2]),
            )
        }
        super().__init__(trial=trial, config=config)



if __name__ == '__main__':

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5)

    sampler = optuna.samplers.RandomSampler()

    tuner = Tuner(
        sampler=sampler,
        pruner=pruner,
        )

    tuner.tune(n_trials=50)

    tuner.xval()

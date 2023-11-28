import optuna

from models.lstm import LSTM
from utils.tuning import SearchSpace, Tuner


class LSTMSearchSpace(SearchSpace):
    """Defines the search space for the LSTM model."""

    MODEL = LSTM

    def __init__(self, trial: optuna.Trial) -> None:
        config = {
            'model': dict(
                num_hidden=trial.suggest_categorical('num_hidden', [32, 64, 128]),
                num_enc=trial.suggest_categorical('num_enc', [4, 8, 16, 32]),
                enc_dropout=trial.suggest_categorical('enc_dropout', [0.0, 0.2]),
                num_enc_layers=trial.suggest_categorical('num_enc_layers', [1, 2, 3]),
                num_lstm_layers=trial.suggest_categorical('num_lstm_layers', [1, 2]),
            ),
            'optimizer': dict(
                lr=trial.suggest_categorical('lr', [1e-3, 1e-2, 1e-1]),
                weight_decay=trial.suggest_categorical('weight_decay', [1e-1, 1e-2, 1e-3]),
            )
        }
        super().__init__(trial=trial, config=config)


if __name__ == '__main__':

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=10,
        reduction_factor=2
    )
    sampler = optuna.samplers.TPESampler(
        consider_prior=False,
        n_startup_trials=10,
        seed=1,
        multivariate=True
    )

    tuner = Tuner(
        sampler=sampler,
        pruner=pruner,
        )

    tuner.tune(n_trials=80)

    tuner.xval()
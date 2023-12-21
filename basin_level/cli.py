import optuna

from models.base import TCN, LSTM, MHA
from utils.tuning import SearchSpace, Tuner


def get_enc_search_space(trial: optuna.Trial) -> dict:

    config = dict(
        model_dim=trial.suggest_categorical('model_dim', [64, 128, 256]),
        enc_dropout=trial.suggest_categorical('enc_dropout', [0.0, 0.2]),
        enc_layers=trial.suggest_categorical('enc_layers', [1, 2]),
    )

    return config

def get_optimizer_search_space(trial: optuna.Trial) -> dict:

    config = dict(
        lr=trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2]),
        weight_decay=trial.suggest_categorical('weight_decay', [1e-1, 1e-2, 1e-3]),
    )

    return config


class LSTMSearchSpace(SearchSpace):
    """Defines the search space for the lonh short-term memory (LSTM) model."""

    MODEL = LSTM

    def __init__(self, trial: optuna.Trial) -> None:
        config = {
            'model': dict(
                **get_enc_search_space(trial),
                lstm_layers=trial.suggest_categorical('lstm_layers', [1, 2]),
            ),
            'optimizer': dict(
                **get_optimizer_search_space(trial),
            )
        }
        super().__init__(trial=trial, config=config)


class TCNSearchSpace(SearchSpace):
    """Defines the search space for the temporal convolutional network (TCN) model."""

    MODEL = TCN

    def __init__(self, trial: optuna.Trial) -> None:
        config = {
            'model': dict(
                **get_enc_search_space(trial),
                tcn_kernel_size=trial.suggest_categorical('tcn_kernel_size', [8, 16]),
                tcn_layers=trial.suggest_categorical('tcn_layers', [2, 3, 4]),
                tcn_dropout=trial.suggest_categorical('tcn_dropout', [0.0, 0.2]),
            ),
            'optimizer': dict(
                **get_optimizer_search_space(trial),
            )
        }
        super().__init__(trial=trial, config=config)


class MHASearchSpace(SearchSpace):
    """Defines the search space for the multihead attention (MHA) model."""

    MODEL = MHA

    def __init__(self, trial: optuna.Trial) -> None:
        config = {
            'model': dict(
                **get_enc_search_space(trial),
                mha_nhead=trial.suggest_categorical('mha_nhead', [2, 4, 8]),
                mha_layers=trial.suggest_categorical('mha_layers', [1, 2, 3]),
                mha_dropout=trial.suggest_categorical('mha_dropout', [0.0, 0.1, 0.2]),
            ),
            'optimizer': dict(
                **get_optimizer_search_space(trial),
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

    tuner.tune(n_trials=20)

    tuner.xval()

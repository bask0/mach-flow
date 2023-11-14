from argparse import ArgumentParser
import torch
from torch import Tensor

from utils.pl import LightningNet
from utils.torch import Normalize, Transform
from utils.torch_modules import FeedForward
from project.machflowdata import MachFlowDataModule


class LSTM(LightningNet):

    def __init__(
            self,
            num_static_in: int,
            num_dynamic_in: int,
            num_enc: int = 4,
            enc_dropout: float = 0.1,
            num_enc_layers: int = 1,
            num_hidden: int = 32,
            num_lstm_layers: int = 1,
            num_outputs: int = 1,
            learning_rate: float = 1e-3,
            weight_decay: float = 0.0,
            norm_args_features: dict | None = None,
            norm_args_stat_features: dict | None = None) -> None:

        super().__init__()

        self.save_hyperparameters(ignore=['norm_args_features', 'norm_args_stat_features'])
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if norm_args_features is not None:
            self.norm_features = Normalize(**norm_args_features)
        else:
            self.norm_features = torch.nn.Identity()

        if norm_args_stat_features is not None:
            self.norm_stat_features = Normalize(**norm_args_stat_features)
        else:
            self.norm_stat_features = torch.nn.Identity()

        if norm_args_stat_features is not None:
            self.norm_targets = Normalize(**norm_args_stat_features)
        else:
            self.norm_targets = torch.nn.Identity()

        self.static_encoding = FeedForward(
            num_inputs=num_static_in,
            num_outputs=num_enc,
            num_hidden=num_hidden,
            num_layers=num_enc_layers,
            dropout=enc_dropout,
            activation='tanh',
            activation_last='tanh',
            dropout_last=True,
            residual=False
        )

        self.to_channels_last = Transform(transform_fun=lambda x: x.transpose(1, 2))

        self.dynamic_encoding = FeedForward(
            num_inputs=num_dynamic_in,
            num_outputs=num_enc,
            num_hidden=num_hidden,
            num_layers=num_enc_layers,
            dropout=enc_dropout,
            activation='tanh',
            activation_last='tanh',
            dropout_last=True,
            residual=False
        )

        self.lstm = torch.nn.LSTM(
            input_size=num_enc,
            hidden_size=num_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        self.out_layer = torch.nn.Linear(
            in_features=num_hidden,
            out_features=num_outputs
        )

        self.out_activation = torch.nn.Softplus()

        self.to_sequence_last = Transform(transform_fun=lambda x: x.transpose(1, 2))

    def forward(self, x: Tensor, s: Tensor | None) -> Tensor:

        # Normalize dynamic inputs with #C features: (B, C, S)
        x_out = self.norm_features(x)

        # Dyncmic inputs to channel last: (B, C, S) -> (B, S, C)
        x_out = self.to_channels_last(x_out)

        # Dynamic encoding: (B, S, C) -> (B, S, E)
        x_out = self.dynamic_encoding(x_out)

        # Normalize static inputs with #D features: (B, C)
        s_out = self.norm_stat_features(s)

        if s is not None:
            # Static encoding and unsqueezing: (B, C) ->  (B, 1, E)
            s_out = self.static_encoding(s_out).unsqueeze(1)
            # Add static encoding to dynamic encoding: (B, S, E) + (B, 1, E) -> (B, S, E)
            x_out = x_out + s_out

        # LSTM layer: (B, S, E) -> (B, S, H)
        out, _ = self.lstm(x_out)

        # Output layer and activation: (B, S, H) -> (B, S, O)
        out = self.out_layer(out)
        out = self.out_activation(out)

        # Dyncmic outputs to sequence last: (B, S, O) -> (B, O, S)
        out = self.to_sequence_last(out)

        return out

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            '--num_static_in', type=int, required=True,
            help='Number of static inputs.'
        )
        parser.add_argument(
            '--num_dynamic_in', type=int, required=True,
            help='Number of dynamic inputs.'
        )
        parser.add_argument(
            '--num_enc', type=int, default=4,
            help='Inputs encoding size.'
        )
        parser.add_argument(
            '--enc_dropout', type=float, default=0.1,
            help='Encoding dropout.'
        )
        parser.add_argument(
            '--num_enc_layers', type=int, default=1,
            help='Number of encoding layers.'
        )
        parser.add_argument(
            '--num_hidden', type=int, default=32 ,
            help='Number of encoding and LSTM hidden nodes.'
        )
        parser.add_argument(
            '--num_lstm_layers', type=int, default=1,
            help='Number of LSTM layers.'
        )
        parser.add_argument(
            '--num_outputs', type=int, default=1,
            help='Number of outputs.'
        )
        parser.add_argument(
            '--learning_rate', type=float, default=1e-3,
            help='Learning rate.'
        )
        parser.add_argument(
            '--weight_decay', type=float, default=0.0,
            help='Weight decay.'
        )

        return parser

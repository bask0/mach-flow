
import torch
from torch import Tensor

from utils.pl import LightningNet
from utils.torch import Normalize
from utils.torch_modules import PadTau, TemporalConvNet, EncodingModule


class TCN(LightningNet):
    """TCN based rainfall-runoff module."""
    def __init__(
            self,
            num_static_in: int,
            num_dynamic_in: int,
            num_enc: int = 16,
            enc_dropout: float = 0.0,
            num_enc_layers: int = 1,
            num_hidden: int = 32,
            kernel_size: int = 4,
            tcn_dropout:float = 0.0,
            num_tcn_layers: int = 1,
            num_outputs: int = 1,
            norm_args_features: dict | None = None,
            norm_args_stat_features: dict | None = None,
            norm_args_stat_targets: dict | None = None,
            **loss_kwargs) -> None:

        super().__init__(**loss_kwargs)

        self.save_hyperparameters()

        # Data normalization
        # -------------------------------------------------

        if norm_args_features is not None:
            self.norm_features = Normalize(**norm_args_features)
        else:
            self.norm_features = torch.nn.Identity()

        if norm_args_stat_features is not None:
            self.norm_stat_features = Normalize(**norm_args_stat_features)
        else:
            self.norm_stat_features = torch.nn.Identity()

        if norm_args_stat_targets is not None:
            self.norm_targets = Normalize(**norm_args_stat_targets)
        else:
            self.norm_targets = torch.nn.Identity()

        # Static input encoding
        # -------------------------------------------------

        self.static_encoding = EncodingModule(
            num_in=num_static_in,
            num_enc=num_enc,
            num_layers=num_enc_layers,
            dropout=enc_dropout,
            activation=torch.nn.ReLU(),
            activation_last=torch.nn.Sigmoid()
        )

        # Dynamic input encoding
        # -------------------------------------------------

        self.dynamic_encoding = EncodingModule(
            num_in=num_dynamic_in,
            num_enc=num_enc,
            num_layers=num_enc_layers,
            dropout=enc_dropout,
            activation=torch.nn.Tanh(),
            activation_last=torch.nn.Tanh()
        )

        # Temporal layer(s)
        # -------------------------------------------------

        self.pad_tau = PadTau()

        self.tcn = TemporalConvNet(
            num_inputs=num_enc + 1,
            num_outputs=-1,
            num_hidden=num_hidden,
            kernel_size=kernel_size,
            num_layers=num_tcn_layers,
            dropout=tcn_dropout
        )

        self.out_layer = torch.nn.Conv1d(
            in_channels=num_hidden + 1,
            out_channels=num_outputs,
            kernel_size=1
        )

        self.out_activation = torch.nn.Softplus()

    def forward(self, x: Tensor, s: Tensor | None, tau: float = 0.5) -> Tensor:

        # Normalize dynamic inputs with #C features: (B, C, S)
        x_out = self.norm_features(x)

        # Dynamic encoding: (B, C, S) -> (B, E, S)
        x_out = self.dynamic_encoding(x_out)

        if s is not None:
            # Normalize static inputs with #D features: (B, C)
            s_out = self.norm_stat_features(s)

            # Static encoding and unsqueezing: (B, C) ->  (B, E, 1)
            s_out = self.static_encoding(s_out.unsqueeze(-1))

            # Add static encoding to dynamic encoding: (B, E, S) + (B, E, 1) -> (B, E, S)
            x_out = x_out + s_out

        # Pad tau: (B, E, S) -> (B, E + 1, S)
        x_out = self.pad_tau(x_out, tau)

        # TCN layer: (B, E + 1, S) -> (B, H, S)
        out = self.tcn(x_out)

        # Pad tau: (B, H, S) -> (B, H + 1, S)
        out = self.pad_tau(out, tau)

        # Output layer and activation: (B, H + 1, S) -> (B, O, S)
        out = self.out_layer(out)

        out = self.out_activation(out)

        return out

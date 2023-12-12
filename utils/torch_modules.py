# some code is adapted from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
# - Chomp1d
# - Residual
# - TemporalBlock
# - TemporalConvNet
# Original license applies.

import torch
from torch import Tensor
import torch.nn.utils.weight_norm as weight_norm

from utils.torch import get_activation
from utils.torch import Transform


class FeedForwardBlock(torch.nn.Module):
    """Implements a FeedForward block.
    """
    def __init__(
            self,
            num_inputs: int,
            num_outputs: int,
            dropout: float,
            activation: str | None,
            batch_norm: bool = False) -> None:
        """Initialize a FeedForward block.

        Args:
            num_inputs (int): input dimensionality
            num_outputs (int): output dimensionality
            dropout (float): dropout applied after each HIDDEN layer, in range [0, 1)
            activation (str, optional): activation function, see `get_activation`
            residual (bool): whether to add a residual connection, default is `False`.
            batch_norm (bool): wheter to use batch normalizations in all but the
                last layer. Defaults to `False`.
        """
        super().__init__()

        self.block = torch.nn.Sequential()

        self.block.add_module('linear', torch.nn.Linear(num_inputs, num_outputs))
        if batch_norm:
            self.block.add_module('batch_norm', torch.nn.BatchNorm1d(num_outputs))
        if dropout > 0.0:
            self.block.add_module('dropout', torch.nn.Dropout(dropout))
        self.block.add_module('activation', get_activation(activation))

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class EncodingModule(torch.nn.Module):
    """Implements an encoding module based on Conv1D layers.

    Each hidden layer consists of a linear, dropout and an activation layer. The last layer does not have
    dropout. The hidden size is `hidden_size_factor` times the encoding size (num_enc).

    Shapes:
        - Input tensor: (B, I, L)
        - Output tensor: (B, E, L)
        ...where B is the batch size, I the input size, E the encoding size, and L the lengths.

    """
    def __init__(
            self,
            num_in: int,
            num_enc: int,
            num_layers: int,
            dropout: float,
            hidden_size_factor: int = 2,
            activation: torch.nn.Module = torch.nn.Sigmoid(),
            activation_last: torch.nn.Module = torch.nn.Sigmoid()) -> None:
        """Initialize EncodingModule

        Args:
            num_in (int): the input channel size (C_in).
            num_enc (int): the encoding size.
            num_layers (int): the number of layers (C_out).
            dropout (float): the dropout applied to all but the output layer.
            unsqueeze_input (bool): whether to unsqueeze the input.
            hidden_size_factor (int): the hidden size as a factor of the encoding size (`num_enc`);
                `hidden_size = hidden_size_factor * num_enc`. The default is 2.
            activation (torch.nn.Module, optional): The activation of hidden layers.
                Defaults to torch.nn.Sigmoid().
            activation_last (torch.nn.Module, optional): the activation of the output layer.
                Defaults to torch.nn.Sigmoid().
        """        
        super().__init__()

        num_hidden = num_enc * hidden_size_factor

        layers: list[torch.nn.Module] = []

        for layer in range(num_layers):
            is_first = layer == 0
            is_last = layer == num_layers - 1

            conv = torch.nn.Conv1d(
                in_channels=num_in if is_first else num_hidden,
                out_channels=num_enc if is_last else num_hidden,
                kernel_size=1,
            )
            layers.append(conv)

            if is_last:
                layers.append(activation_last)
            else:
                layers.append(torch.nn.Dropout(dropout))
                layers.append(activation)

        self.encoding = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoding(x)


class FeedForward(torch.nn.Module):
    """Implements a feed-forward neural networks with 1 to n layers.

    Each layer conisting of a linear, dropout and an activation layer. The dropout and activation of the
    last layer are optional (see `activation_last` and `dropout_last`). Optionally adds a redidual connection 
    with linear transformation from the input to the oputput.

    Shapes:
        - Input tensor: (B, ..., I)
        - Output tensor: (B, ..., O)
        ...where I is num_inputs and O is num_outputs, and B is the batch size.

    """
    def __init__(
            self,
            num_inputs: int,
            num_outputs: int,
            num_hidden: int,
            num_layers: int,
            dropout: float,
            activation: str,
            activation_last: str | None = None,
            dropout_last: bool = False,
            residual: bool = False,
            batch_norm: bool = False) -> None:
        """Initialize a feed-forward neural networks with 1 to n layers.

        Args:
            num_inputs (int): input dimensionality
            num_outputs (int): output dimensionality
            num_hidden (int): number of hidden units
            num_layers (int): number of fully-connected hidden layers (not
                incliding the input and output layers)
            dropout (float): dropout applied after each HIDDEN layer, in range [0, 1)
            activation (str): activation function, see `get_activation`
            activation_last (str, optional): output activation, see `get_activation`.
                Defaults to None (=identity).
            dropout_last (bool, optional): if `True`, the dropout is also
                applied after last layer. Defaults to False.
            residual (bool): whether to add a residual connection, default is `False`.
            batch_norm (bool): wheter to use batch normalizations in all but the
                last layer. Defaults to `False`.

        """
        super().__init__()

        self.residual = residual
        if self.residual:
            self.residual_transform = torch.nn.Linear(num_inputs, num_hidden)

        self.layer_in = FeedForwardBlock(
            num_inputs=num_inputs,
            num_outputs=num_hidden,
            dropout=0.0,
            activation=activation,
            batch_norm=batch_norm
        )

        self.layers_hidden = torch.nn.ModuleDict()
        for idx in range(num_layers):
            layer_hidden = FeedForwardBlock(
                num_inputs=num_hidden,
                num_outputs=num_hidden,
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm
            )
            self.layers_hidden.update({f'layer_h{idx:02d}': layer_hidden})

        self.layer_out = FeedForwardBlock(
            num_inputs=num_hidden,
            num_outputs=num_outputs,
            dropout=dropout if dropout_last else 0.0,
            activation=activation_last,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Model forward call.

        Args:
            x (Tensor): the input tensor.

        Returns:
            Tensor: the output tensor.
        """

        if self.residual:
            x_res = self.residual_transform(x)
        else:
            x_res = 0.0

        out = self.layer_in(x)

        for _, layer in self.layers_hidden.items():
            out = layer(out)

        out = out + x_res

        out = self.layer_out(out)

        return out


class Chomp1d(torch.nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class Residual(torch.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        """Residual connection, does downsampling if necessary.

        Args:
            n_inputs: layer input size (number of channels).
            n_outputs: layer output size (number of channels).
        """
        super(Residual, self).__init__()
        self.do_downsample = n_inputs != n_outputs
        self.downsample = torch.nn.Conv1d(n_inputs, n_outputs, 1) if self.do_downsample else torch.nn.Identity()

    def forward(self, x: Tensor, res: Tensor, s_res: Tensor | None = None) -> Tensor:
        """Residual connection.

        Shapes:
            x: (batch_size, num_outputs, sequence_length)
            res: (batch_size, num_inputs, sequence_length)
            s_res: (batch_size, num_static_inputs)
            output: (batch_size, num_outputs, sequence_length)

        Args:
            x: the input.
            res: the residual input.
            s_res: the optional static residual input.

        Returns:
            The output tensor.

        """
        if s_res is None:
            return x + self.downsample(res)
        else:
            res = torch.cat(
                (
                    res,
                    s_res.unsqueeze(-1).expand(s_res.shape[0], s_res.shape[1], res.shape[-1])
                ),
                dim=-2
            )

            return x + self.downsample(res)

    def init_weights(self) -> None:
        if self.do_downsample:
            self.downsample.weight.data.normal_(0, 0.01)


class TemporalBlock(torch.nn.Module):
    def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            kernel_size: int,
            stride: int,
            dilation: int,
            padding: int,
            dropout: float = 0.2) -> None:
        """Implements a two-layered residual block.

        Args:
            n_inputs: number of inputs (channels).
            n_outputs: number of outputs (channels).
            kernel_size: the 1D convolution kernel size.
            stride: the 1D convolution stride.
            dilation: the 1D convolution dilation.
            padding: the padding.
            dropout: the dropout applied after each layer. Defaults to 0.2.
        """

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            torch.nn.Conv1d(
                n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = torch.nn.Softplus()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.conv2 = weight_norm(
            torch.nn.Conv1d(
                n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = torch.nn.Softplus()
        self.dropout2 = torch.nn.Dropout(dropout)

        self.res = Residual(n_inputs, n_outputs)
        self.act = torch.nn.Softplus()
        self.init_weights()

    def init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        self.res.init_weights()

    def forward(self, x: Tensor, s: Tensor | None) -> Tensor:
        """Model forward run.

        Shapes:
            Input x:  (batch_size, x_input_size, sequence_length)
            Input s:  (batch_size, s_input_size)
            Output:   (batch_size, num_outputs, sequence_length)

        Args:
            x: the input dynamic tensor.
            s: the input static tensor. None (default) means no static inputs. Then,
                model parameter `n_static_inputs` must be `0`.

        Returns:
            The output tensor.
        """

        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.act2(out)
        out = self.dropout2(out)

        out = self.res(out, x, s)
        return self.act(out)


class TemporalConvNet(torch.nn.Module):
    def __init__(
            self,
            num_inputs: int,
            num_outputs: int,
            num_hidden: int,
            kernel_size: int = 4,
            num_layers: int = 2,
            dropout: float = 0.0) -> None:
        """Implements a Temporal Convolutional Network (TCN).

        https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

        Note:
            The TCN layer is followed by a feedfoward layer to map the TCN output channels to `num_outputs`.

        Shapes:
            Input: (batch_size, num_inputs, sequence_length)
            Startic input: (batch_size, num_static_inputs)
            Output: (batch_size, num_outputs, sequence_length)

        Args:
            num_inputs: the number of input features.
            num_static_inputs: the number of static input features.
            num_outputs: the number of outputs. If -1, no linear mapping is done.
            num_hidden: the hidden size (intermediate channel sizes) of the layers.
            kernel_size: the kernel size. Defaults to 4.
            num_layers: the number of stacked layers. Defaults to 2.
            dropout: a float value in the range [0, 1) that defines the dropout probability. Defaults to 0.0.
        """

        super().__init__()

        # Used to calculate receptive field (`self.receptive_field_size`).
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.tcn_layers = torch.nn.ModuleDict()
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_hidden
            layer = TemporalBlock(
                n_inputs=in_channels,
                n_outputs=num_hidden,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout
            )

            self.tcn_layers.update({f'layer_h{i:02d}': layer})

        if num_outputs == -1:
            self.linear = torch.nn.Identity()
        else:
            self.linear = torch.nn.Conv1d(
                in_channels=num_hidden,
                out_channels=num_outputs,
                kernel_size=1
            )

    def forward(self, x: Tensor) -> Tensor:
        """Run data through the model.

        Args:
            x: the input data with shape (batch, num_inputs, seq).
            s: the optional static input data with shape (batch, num_static_inputs).

        Returns:
            the model output tensor with shape (batch, seq, num_outputs).
        """

        out = x
        for _, layer in self.tcn_layers.items():
            out = layer(out, None)

        out = self.linear(out)
        return out

    def receptive_field_size(self) -> int:
        """Returns the receptive field of the Module.

        The receptive field (number of steps the model looks back) of the model depends
        on the number of layers and the kernel size.

        Returns:
            the size of the receptive field.

        """

        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.num_layers - 1)

    def __repr__(self):
        receptive_field = self.receptive_field_size()
        s = super().__repr__() + f' [context_len={receptive_field}]'
        return s


class PadTau(torch.nn.Module):
    """Implements padding for qiuantile and expectile losses.

    Padds the value `tau` to a 3D tensor at the end of the channel dimension (second one).
    
    Shapes:
        x: (batch, channels, sequence)
        out: (batch, channels + 1, sequence)
    """
    def __init__(self): 
        """Initialize PadTau."""
        super().__init__()

    def forward(self, x: Tensor, tau: float) -> Tensor:
        if (ndim := x.ndim) != 3:
            raise ValueError(
                f'input must have three dimensions (batch, channel, sequence), has {ndim}.'
            )

        return torch.nn.functional.pad(x, (0, 0, 0, 1, 0, 0), mode='constant', value=tau)
    
import torch
from torch import Tensor
import numpy as np
import xarray as xr

from typing import Any, Callable


TensorLike = np.ndarray[int, np.dtype[np.float32]] | torch.Tensor


class Normalize(torch.nn.Module):
    """Normalization module.

    The module is initialized with precomputed means and standard deviations, which are then used to normalize the
    input data on-the-fly. Optionally, the inverse (denormalization) can be computed.

    Note that the data statistics should be computed from the training set!

    Equations:
        - normalization: (x - mean(x_global)) / std(x_global)
        - de normalization: x * std(x_global) + std(x_global)

    Shapes:
        In the forward call, the input tensor is expected to be batched. For sample-level normalization,
        use the `normalize` and `denormalize` methods, respectively. The argument `features_dim` indicates the
        channel dimension to normalized without the batch dimension. For an input with shape (B, C, S), with the
        channel dimension at position 1, the features_dim would be 0 (corresponding to the unbatched data).

        - Input shape: (B, ..., C, ...)
        - Output shape: (B, ..., C, ...)

        ...where C must correspond to the provided `means` and `stds` vector length.

    """    
    def __init__(
            self,
            means: TensorLike,
            stds: TensorLike,
            features_dim: int = 0,
            dtype: torch.dtype = torch.float32) -> None:
        """Initialize the Normelize module.

        Args:
            means (TensorLike): A vector of mean values for normalization.
            stds (TensorLike): A vector of standard deviations for normalization.
            features_dim (int, optional): The dimension to normalize over. Corresponds to unbatched dimension of the
                sample. If a sample has shape (channels=4, sequence=100), and (batch=10, channels=4, sequence=100), we
                want to normalize over dim=0, as this is the feature dimension. Defaults to 0.
            dtype (torch.dtype, optional): The data dtype. Defaults to torch.float32.
        """
        super().__init__()

        self.means = torch.as_tensor(means, dtype=dtype)
        self.stds = torch.as_tensor(stds, dtype=dtype)
        self.dim = features_dim

    def forward(self, x: Tensor, invert: bool = False) -> Tensor:
        """(De)normalize input tensor `x` using predefined statistics.

        Args:
            x (Tensor): The input tensor to be normalized.
            invert (bool, optional): Whether to normalize (invert=False) or denormalize (invert=True).
                Defaults to False.

        Returns:
            Tensor: The (de)normalized tensor `x`.
        """        
        if invert:
            return self.denormalize(x, is_batched=True)
        else:
            return self.normalize(x, is_batched=True)

    def normalize(self, x: Tensor, is_batched: bool = False) -> Tensor:
        """Normalize input x.

        Equation: (x - mean(x_global)) / std(x_global) 

        Args:
            x (Tensor): The input tensor to normalize. The stats (`means` and `stds`) are applied over dimension
                `dim`, or `dim` + 1 if `is_batched=True`.
            is_batched (bool, optional): Whether the input Tensor x is a batch (first dim is batch dim) or a single
                sample. Defaults to False.

        Returns:
            Tensor: The normalized tensor with same shape as input.
        """        
        means, stds = self._expand_stats_to_x(x=x, is_batched=is_batched)

        return (x - means) / stds

    def denormalize(self, x: Tensor, is_batched: bool = False) -> Tensor:
        """Denormalize input x.

        Equation: x * std(x_global) + std(x_global) 

        Args:
            x (Tensor): The input tensor to normalize. The stats (`means` and `stds`) are applied over dimension
                `dim`, or `dim` + 1 if `is_batched=True`.
            is_batched (bool, optional): Whether the input Tensor x is a batch (first dim is batch dim) or a single
                sample. Defaults to False.

        Returns:
            Tensor: The denormalized tensor with same shape as input.
        """      
        means, stds = self._expand_stats_to_x(x=x, is_batched=is_batched)

        return x * stds + means

    @staticmethod
    def get_normalize_args(
            ds: xr.Dataset,
            norm_variables: str | list[str],
            dtype: torch.dtype = torch.float32) -> dict[str, Any]:

        norm_variables = [norm_variables] if isinstance(norm_variables, str) else norm_variables

        means = []
        stds = []
        for var in norm_variables:
            da = ds[var].load()
                
            means.append(da.mean().item())
            stds.append(da.std().item())

        args = {
            'means': torch.as_tensor(means, dtype=dtype),
            'stds': torch.as_tensor(stds, dtype=dtype),
            'features_dim': 0,
            'dtype': dtype
        }

        return args

    def _expand_stats_to_x(self, x: Tensor, is_batched: bool) -> tuple[Tensor, Tensor]:
        """Expand (aka unsqueeze) stats to match input TensorLike shape.
        
        x (TensorLike): The Tensor to (de)normalize.
        is_batched (bool): Whether the input Tensor x is a batch (first dim is batch dim) or a single sample.
        """
        dim = self.dim + 1 if is_batched else self.dim
        dims = tuple([slice(None, None) if d == dim else None for d in range(x.ndim)])

        means_expanded = self.means[dims]
        stds_expanded = self.stds[dims]

        return means_expanded, stds_expanded


def get_activation(activation: str | None) -> torch.nn.Module:
    """Get PyTorch activation function by string query.

    Args:
        activation (str, None): activation function, one of `relu`, `leakyrelu`, `selu`, `sigmoid`, `softplus`,
            `tanh`, `identity` (aka `linear` or `none`). If `None` is passed, `None` is returned.

    Raises:
        ValueError: if activation not found.

    Returns:
        PyTorch activation or None: activation function
    """

    if activation is None:
        return torch.nn.Identity()

    a = activation.lower()

    if a == 'linear' or a == 'none':
        a = 'identity'

    activations = dict(
        relu=torch.nn.ReLU,
        leakyrelu=torch.nn.LeakyReLU,
        selu=torch.nn.SELU,
        sigmoid=torch.nn.Sigmoid,
        softplus=torch.nn.Softplus,
        tanh=torch.nn.Tanh,
        identity=torch.nn.Identity
    )

    if a in activations:
        return activations[a]()
    else:
        choices = ', '.join(list(activations.keys()))
        raise ValueError(
            f'activation `{activation}` not found, chose one of: {choices}.'
        )


class Transform(torch.nn.Module):
    def __init__(self, transform_fun: Callable) -> None:
        """Transform layer, applies `transform_fun`.

        Example:
            >>> import torch
            >>> reshape_fun = lambda x: x.flatten(start_dim=1)
            >>> flatten_layer = Transform(reshape_fun)
            >>> flatten_layer(torch.ones(2, 3, 4)).shape
            torch.Size([2, 12])

        Args:
            transform_fun (Callable): thetransform function.
        """
        super(Transform, self).__init__()

        self.transform_fun = transform_fun

    def forward(self, x: Tensor) -> Tensor:
        return self.transform_fun(x)



import torch
from torch import nn
from torch import Tensor
import math
import logging

logger = logging.getLogger('pytorch_lightning')


class BetaNLLLoss(nn.Module):

    def __init__(self, reduction: str = 'mean', beta: float = 0.5) -> None:
        r"""Creates beta-NLL criterion

        The beta-NLL criterion extends the standard Gaussian NLL criterion. The beta
        parameter specifies a weight fo the errors from 0 to 1, 0 yielding standard NLL
        criterion, and 1 weighing as in MSE (but still with uncertainty). This solves a
        problem with standard NLL, which tends to emphasize regions with low variance
        during trainging. See https://arxiv.org/pdf/2203.09168.pdf

        Notes:
            When beta is set to 0, this loss is equivalent to the NLL loss. Beta of 1 corresponds
            to equal weighting of low and high error points.

        Args:
            reduction: Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
            beta: Parameter from range [0, 1] controlling relative weighting between data points,
                where `0` corresponds to high weight on low error points and `1` to an equal weighting.
                The default, 0.5, has been reported to work best in general.

        Shape:
            - mean: (batch_size, ...).
            - variance: (batch_size, ...), same shape as the mean.
            - target: (batch_size, ...), same shape as the mean.
            - output: scalar, or, if `reduction` is ``'none'``, then (batch_size, ...), same shape as the input.
        """
        super().__init__()

        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(
                f'argument `reduction` must be one of (\'none\', \'mean\', \'sum\'), is {reduction}.'
            )

        if not (0 <= beta <= 1):
            raise ValueError(
                f'`beta` must be a value in the range [0, 1], is {beta}.'
            )

        self.reduction = reduction
        self.beta = beta

        self._FILL_NLL = None
        self._FILL_BETA = None

    def forward(
            self,
            mean: Tensor,
            variance: Tensor,
            target: Tensor,
            mask: Tensor | None = None) -> Tensor:
        """Compute beta-NLL loss (https://arxiv.org/pdf/2203.09168.pdf)

        Args:
            mean: Predicted mean of shape B x D
            variance: Predicted variance of shape B x D
            target: Target of shape B x D
            mask: A mask indicating valid (i.e., finite and non-NaN) elements in the `target`.
                `None` (default), implies that no missing values are present in `target`.

        Returns:
            Loss per batch element of shape B
        """

        # This value replaces NaN in the variance, yields aproximately 0 from gaussian_nll_loss.
        if self._FILL_NLL is None:
            self._FILL_NLL = torch.tensor(1 / (2 * math.pi), requires_grad=False, device=mean.device, dtype=mean.dtype)
        # This value replaces NaN in the variance, yields 0 from beta weighing.
        if self._FILL_BETA is None:
            self._FILL_BETA = torch.tensor(0.0, requires_grad=False, device=mean.device, dtype=mean.dtype)

        if mask is not None:
            variance = variance.where(mask, self._FILL_NLL)

        loss = nn.functional.gaussian_nll_loss(
            input=mean,
            target=target,
            var=variance,
            full=True,
            reduction='none'
        )

        if self.beta > 0.0:
            if mask is not None:
                variance = variance.where(mask, self._FILL_BETA)
            loss = loss * variance.detach() ** self.beta

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class BaseQLoss(nn.Module):
    def __init__(self, reduction='none') -> None:
        r"""Creates element-wise distribution-based loss without reduction. Must be subclassed.

        Note: The error function `error_function` mut be overridden in the subclass.

        Shape:
            - mean: (batch_size, ...).
            - target: (batch_size, ...), same shape as the mean.
            - output: (batch_size, ...), same shape as the mean.

        Args:
            reduction: for compatibility, has no effect.
        """
        super().__init__()

    def forward(
            self,
            input: Tensor,
            target: Tensor,
            tau: float | Tensor) -> Tensor:
        """Compute the losss.

        Args:
            input: Predicted mean of shape (B, ...)
            target: Target of shape (B, ...)
            tau: The probability to evaluate, in range [0, 1].

        Returns:
            Loss per batch element of shape (B, ...)
        """

        if not (0.0 <= tau <= 1.0):
            raise ValueError(
                f'argument `tau` is out of range [0, 1] with `tau`={tau}`'
            )

        err = self.error_function(input=input, target=target, tau=tau)

        return err

    def error_function(self, input: Tensor, target: Tensor, tau: float | Tensor) -> Tensor:
        raise NotImplementedError(
            '`err_fun` must be defined in subclass.'
        )


class QLoss(BaseQLoss):

    def __init__(self, reduction='none') -> None:
        r"""Creates element-wise quantile loss without reduction.

        Shape:
            - mean: (batch_size, ...).
            - target: (batch_size, ...), same shape as the mean.
            - output: (batch_size, ...), same shape as the mean.
        """
        super().__init__()

    def error_function(self, input: Tensor, target: Tensor, tau: float | Tensor) -> Tensor:
        err = target - input
        err = torch.max((tau - 1) * err, tau * err)
        return err


class ELoss(BaseQLoss):

    def __init__(self, reduction='none') -> None:
        r"""Creates element-wise expectile loss without reduction.

        Shape:
            - mean: (batch_size, ...).
            - target: (batch_size, ...), same shape as the mean.
            - output: (batch_size, ...), same shape as the mean.
        """
        super().__init__()

    def error_function(self, input: Tensor, target: Tensor, tau: float | Tensor) -> Tensor:
        err = target - input
        err2 = err ** 2
        return torch.where(err >= 0.0, err2 * tau, err2 * (1 - tau))


class NSELoss(nn.Module):

    def __init__(self, reduction: str = 'mean', constant: float = 0.1) -> None:
        r"""Creates NSE* criterion (not NSE, see details below)

        NSE* is based on the Nash-Sutcliffe Modelling Efficiency Coefficient. The NSE* loss is based on
        Kratzert et al. (2019): doi: 10.5194/hess-23-5089-2019

        Instead of ranging from -inf to 1, it ranges from 0 to inf and 0 is the optimum (i.e., we can still use
        minimization).

        It is calculated as `NSE* = 1 / N  (y_hat - y) ** 2 / (s(b) + e) ** 2`, where N is the number of samples,
        s(b) is the standard deviation of the observed basin runoff based on the training period, and e is a constant 
        term used to avoid exploding loss with small basin variance.

        The NSE* is calculated sample-wise first and then averaged across batch elements (samples).

        Notes:
            The NSE above zero indicates better performance than taking the mean of the observations.

        Args:
            reduction: Specifies the reduction to apply to the output. This arguments is used htere for compatibility
                and takes no effect. If another argument than 'mean' is passed, an error is raised. This is because
                the NSE must computed per sample first.
            constant: the constant term added to the basin standard deviation, default is 0.1 as in the source.

        Shape:
            - mean: (batch_size, ...).
            - target: (batch_size, ...), same shape as the mean.
            - output: scalar.
        """
        super().__init__()

        if reduction != 'mean':
            raise ValueError(
                'invalid argument: reduction must be \'mean\'.'
            )

        self.constant = constant

    @staticmethod
    def unsqueeze_like(x: torch.Tensor, ref: torch.Tensor):
        n_unsqueezes = ref.ndim - x.ndim
        if n_unsqueezes == 0:
            return x
        else:
            return x[(...,) + (None,) * n_unsqueezes]

    @staticmethod
    def sigmoid_k(x: Tensor, mu: float, s: float) -> Tensor:
        return (1. / (1. + torch.exp(-(x - mu) / s)))

    def forward(
            self,
            input: Tensor,
            target: Tensor,
            basin_std: Tensor) -> Tensor:
        """Compute the NSE losss.

        Args:
            mean: Predicted mean of shape (B, ...)
            target: Target of shape (B, ...)
            basin_std: The basin standard deviation, a Tensor of shape (B,).

        Returns:
            Loss per batch element of shape B
        """

        mask = target.isfinite()
        red_dims = tuple(range(1, mask.ndim))

        basin_norm = (self.unsqueeze_like(basin_std, mask) + self.constant) ** 2

        target = target.where(mask, input)
        element_err = (target - input) ** 2
        element_err /= basin_norm

        element_num_valid = mask.sum(red_dims)
        element_weight = self.sigmoid_k(x=element_num_valid, mu=100, s=20)

        batch_err = element_err.sum(red_dims) / element_num_valid

        err = (batch_err * element_weight).sum() / element_weight.sum()

        return err



class RegressionLoss(nn.Module):
    """Loss functions that ignore NaN in target.
    The loss funtion allows for having missing / non-finite values in the target.

    !IMPORTANT NOTE!
    ------------------
    With sqrt_transform=True, negative target values are set to 0, and the model predictions are expected to be >= 0.
    This can be achieved by using a activation function such as ReLU or Softplus.

    Example
    -------
    >>> import torch
    >>> mse_loss = RegressionLoss(sample_wise=True)
    >>> input = torch.ones(2, 2, requires_grad=True)
    >>> target = torch.ones(2, 2, requires_grad=False) + 1.
    >>> target[0, 0] = float('NaN')
    >>> loss = mse_loss(input, target)
    >>> loss.backward()
    >>> print('input:', input)
    input:
     tensor([[1., 1.],
             [1., 1.]], requires_grad=True)
    >>> print('target:', target)
    target:
     tensor([[nan, 2.],
             [2., 2.]])
    >>> print('mse:', loss)
    mse:
     tensor(1., grad_fn=<MeanBackward0>)
    >>> print('gradients:', input.grad)
    gradients:
     tensor([[ 0.0000, -1.0000],
             [-0.5000, -0.5000]])

    Shape
    -----
    * input: (N, *), where * means, any number of additional dimensions
    * target: (N, *), same shape as the input
    """

    LOSS_FUNCTIONS = ('l1', 'l2', 'huber', 'nse', 'quantile', 'expectile')
    LOSS_FUNCTIONS_WITH_TAU = ('quantile', 'expectile')

    def __init__(
            self,
            criterion: str,
            sample_wise: bool = True,
            sqrt_transform: bool = False) -> None:
        """Initialize RegressionLoss.

        Args:
            criterion : str (``'l1'`` | ``'l2'`` | ``'huber'`` | ``'nse'`` | ``'quantile'`` | ``'expectile'``)
                ``'l1'`` for Mean Absolute Error (MAE),
                ``'l2'`` for Mean Squared Error (MSE),
                ``'huber'`` for Huber loss (mix of l1 and l2),
                ``'nse'`` for Nash-Sutcliffe modelling efficiency coefficient (NSE),
                ``'quantile'`` for quantile loss,
                ``'expectile'`` for quantile loss,
            sample_wise : bool
                Whether to calculate sample-wise loss first and average then (`True`, default) or to
                calculate the loss across all elements. The former weights each batch element equally,
                the latter weights each observation equally. This is relevant especially with many NaN
                in the target tensor, while there is no difference without NaN.
            sqrt_transform: Whether to sqrt-transform the predictions and targets before computing the loss.
                Note that negative targets are clamped to 0. Default is False.

        """

        super(RegressionLoss, self).__init__()

        criterion = criterion.lower()

        if criterion not in self.LOSS_FUNCTIONS:
            loss_functions_str = '(\'' + '\' | \''.join(self.LOSS_FUNCTIONS) + '\')'
            raise ValueError(
                f'argument `criterion` must be one of {loss_functions_str}, is \'{criterion}\'.'
            )

        self.has_tau = criterion in self.LOSS_FUNCTIONS_WITH_TAU
        self.criterion = criterion
        self.sample_wise = sample_wise

        loss_fn_args: dict = dict(reduction='none')
        if self.criterion == 'huber':
            loss_fn_args.update(dict(delta=0.3))
        elif self.criterion == 'nse':
            loss_fn_args.update(dict(reduction='mean'))
            if not self.sample_wise:
                raise ValueError(
                    'Cannot use criterion \'nse\' with sample_wise being False, as the mean observations '
                    'must be calculated per sample. Use `sample_wise=True` with NSE.'
                )

        self.sqrt_transform = sqrt_transform

        self.loss_fn = {
            'l1': nn.L1Loss,
            'l2': nn.MSELoss,
            'huber': nn.HuberLoss,
            'nse': NSELoss,
            'quantile': QLoss,
            'expectile': ELoss
        }[self.criterion](**loss_fn_args)

    def forward(
            self,
            input: Tensor,
            target: Tensor,
            tau: float | Tensor | None = None,
            basin_std: Tensor | None = None) -> Tensor:
        """Forward call, calculate loss from input and target, must have same shape.

        Args:
            input: Predicted mean of shape (B x ...)
            target: Target of shape (B x ...)
            tau: the probability for quantile or expectile loss, a float in the range [0, 1].
            basin_std: The observation time series standard deviation with shape (B, ). Exclusively required
                for 'nse' criterion.

        Returns:
            The loss, a scalar.
        """

        if self.sqrt_transform:
            input = torch.sqrt(input)
            target = torch.sqrt(target.clamp(min=0))

        if self.criterion == 'nse':
            if basin_std is None:
                raise ValueError(
                    'argument `basin_std` required with `criterion`=\'nse\'.'
                )

            return self.loss_fn(input=input, target=target, basin_std=basin_std)

        mask = target.isfinite()
        # By setting target to input for NaNs, we set gradients to zero.
        target = target.where(mask, input)

        if self.criterion in ['quantile', 'expectile']:
            if tau is None:
                raise ValueError(
                    f'argument `tau` required with `criterion`=\'{self.criterion}\'.'
                )
            element_error = self.loss_fn(input, target, tau)
        else:
            element_error = self.loss_fn(input, target)

        if self.sample_wise:
            red_dims = tuple(range(1, element_error.ndim))
            batch_error = element_error.sum(red_dims) / mask.sum(red_dims)
            err = batch_error.mean()

        else:
            err = element_error.sum() / mask.sum()

        return err

    def __repr__(self) -> str:
        criterion = self.criterion
        sample_wise = self.sample_wise
        sqrt_transform = self.sqrt_transform
        s = f'RegressionLoss({criterion=}, {sample_wise=}, {sqrt_transform=})'
        return s

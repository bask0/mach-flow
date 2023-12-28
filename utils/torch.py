import torch


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

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
        squash_output_sigmoid: bool = False,
        dropout: float = 0,
) -> List[nn.Module]:

    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.
    需要输入和输出的维度，以及中间隐藏层网络神经元个数。nn.Sequential将使用它创建全连接神经网络。
    输入维度是必须的，输出维度可以不必要（例如设置为-1）。不设置输出维度表示输出为最后一层隐藏层的激活值。

    :param squash_output_sigmoid:
    :param dropout:
    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return: List
    """

    # 输入层
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
        if dropout > 0 and len(net_arch) > 0:
            modules.append(nn.Dropout(dropout))
    else:   # no hidden layers
        modules = []

    # 隐藏层
    for idx in range(len(net_arch) - 1):  # 至少有两个隐藏层
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))

    # 输出层
    if output_dim > 0:
        # modules.append(nn.Dropout(dropout))
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim  # no hidden layers
        modules.append(nn.Linear(last_layer_dim, output_dim))

    # add at most one activate function
    assert (not squash_output) or (not squash_output_sigmoid), \
        'at least one of \'squash_output\'/ \'squash_output_sigmoid\' has to be False'

    if squash_output:
        modules.append(nn.Tanh())  # nn module
    if squash_output_sigmoid:
        modules.append(nn.Sigmoid())

    return modules
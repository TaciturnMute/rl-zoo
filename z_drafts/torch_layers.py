import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable
'''
create_mlp:
create_mlp Function is used when creating a sequential Linear Model(MLP). 

DualingPost:
DualingPost Class will create a module acts as a feature extractor,
it does NOT generate some results that can be directly used 
but output a latent representatin for the value net and action network.

This module will include three parts: |shared net : state --> shared_latent
                                      | |policy net : shared_latent --> latnet_pi
                                      | |value net : shared_latent --> latent_vf
                                      
Shared net receives input state and generates shared_latent as input for policy net and value net, 
these two nets receive shared_latent and output latent_pi and latent_vf respectively.
latent_pi and latent_vf can be processed further by other nets. 
For example, latent_pi can used as mean_action of normal distribution in PPO algorithm, 
and a value net can output value estimation after receiving the latent_vf as input.

The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
of them are shared between the policy network and the value network. 
It is assumed to be a list with the following structure:
1. An ARBITRARY length (zero allowed) number of integers each specifying the number of units in a shared layer.
   If the number of ints is zero, there will be no shared layers.
2. An OPTIONAL dict, to specify the following non-shared layers for the value network and the policy network.
   It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
   If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
would be specified as [128, 128].

'''


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



    # if squash_output:
    #     modules.append(nn.Tanh()) # nn module
    #
    # if squash_output_sigmoid:
    #     modules.append(nn.Sigmoid())


    return modules

# net = create_mlp(291,29,[256,256,256])
# net = create_mlp(input_dim=11,output_dim=1,net_arch=[64,64],squash_output=True, dropout=0.2)
# net = create_mlp(input_dim=11,output_dim=-1,net_arch=[64,64],squash_output=True, dropout=0.2)
# net = create_mlp(input_dim=11,output_dim=1,net_arch=[],squash_output=True, dropout=0.2)
# net = create_mlp(input_dim=11,output_dim=-1,net_arch=[64],squash_output=True, dropout=0.2)
# print(nn.Sequential(*net))


"""
DualingPost类的net_arch参数中的layer都表示隐藏层。
DualingPost构造的网络接收特征，输出特征。具体的价值或动作的生成需要在Actor和Critic或Policy中具体写出。
DualingPost的share net和分支net可以是空，表示不存在对应的处理过程，输入直接为输出。

DualingPost_ActorCritic              
            ************************
             ActorCritic shared net
            ************************
                        |
                        |
                  latent feature
            ____________|___________
            |                      |
     ***************         ****************
        Critic Net              Actor Net
     ***************         ****************
            |                       |
     latent_feature         mean_action, log_std
         
    
DualingPost_MeanStd  
            ************************
                Actor shared Net
            ************************
                        |
                        |
                  latent feature
            ____________|___________
            |                      |
     ***************         ***************
     Mean_Action Net           Log_Std Net
     ***************         ***************
            |                      |
        mean_action             log_std

DualingPost_ValueAdvantage
            ************************
                Q-Net shared Net
            ************************
                        |
                        |
                  latent feature
            ____________|___________
            |                      |
     ***************         ***************
        Value Net             Advantage Net
     ***************         ***************
            |                      |
        vf feature             af feature
    

"""

class DualingPost_ActorCritic(nn.Module):
    '''
    policy 表示生成latent_pi的一支
    value 表示生成latent_vf的一支
    '''
    def __init__(self,
                 net_arch: List[Union[int, Dict[str, List[int]]]] = None,
                 input_dim: int = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):
        super(DualingPost_ActorCritic, self).__init__()

        # Iterate through the shared layers and build the shared parts of the network
        shared_net = []
        last_layer_dim_shared_net = input_dim
        policy_only_layers = []
        value_only_layers = []
        for layer in net_arch:
            if isinstance(layer, int):  # Check that this is a shared layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared_net, layer))  # add linear of size layer
                shared_net.append(activation_fn())
                last_layer_dim_shared_net = layer
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        # get value_net from value_only_layers, get policy_net from policy_only layers
        value_net = []
        policy_net = []

        last_layer_dim_value_net = last_layer_dim_shared_net
        last_layer_dim_policy_net = last_layer_dim_shared_net
        # value_only_layers maybe null list
        if len(value_only_layers) > 0:
            last_layer_dim_value_net = last_layer_dim_shared_net  # value net input dim
            for vf_layer_dim in value_only_layers:
                assert isinstance(vf_layer_dim, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_value_net,vf_layer_dim)) # add Linear layer
                value_net.append(activation_fn())  #add activation function
                last_layer_dim_value_net = vf_layer_dim

        if len(policy_only_layers) > 0:
            last_layer_dim_policy_net = last_layer_dim_shared_net  # policy net input dim
            for pi_layer_dim in policy_only_layers:
                assert isinstance(pi_layer_dim, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_policy_net,pi_layer_dim))
                policy_net.append(activation_fn())
                last_layer_dim_policy_net = pi_layer_dim

        self.latent_vf_dim = last_layer_dim_value_net
        self.latent_pi_dim = last_layer_dim_policy_net

        self.shared_net = nn.Sequential(*shared_net)
        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)

    def forward(self,states:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        # policy forward use this
        shared_latent = self.shared_net(states)
        latent_vf = self.value_net(shared_latent)
        latent_pi = self.policy_net(shared_latent)
        return (latent_pi,latent_vf)

    def forward_critic(self,states:torch.Tensor) -> torch.Tensor:
        # only get latent representation for value network
        return self.value_net(self.shared_net(states))

    def forward_actor(self,states:torch.Tensor) -> torch.Tensor:
        # only get latent representation for policy network
        return self.policy_net(self.shared_net(states))

class DualingPost_ValueAdvantage(nn.Module):
    '''

    '''

    def __init__(self,
                 net_arch: List[Union[int, Dict[str, List[int]]]] = None,
                 input_dim: int = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):
        super(DualingPost_ValueAdvantage, self).__init__()

        # Iterate through the shared layers and build the shared parts of the network
        shared_net = []
        last_layer_dim_shared_net = input_dim
        vf_only_layers = []
        af_only_layers = []
        for layer in net_arch:
            if isinstance(layer, int):  # Check that this is a shared layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared_net, layer))  # add linear of size layer
                shared_net.append(activation_fn())
                last_layer_dim_shared_net = layer
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    vf_only_layers = layer["vf"]

                if "af" in layer:
                    assert isinstance(layer["af"], list), "Error: net_arch[-1]['af'] must contain a list of integers."
                    af_only_layers = layer["af"]
                break  # From here on the network splits up in policy and value network

        # get value_net from value_only_layers, get policy_net from policy_only layers
        vf_net = []
        af_net = []

        last_layer_dim_vf_net = last_layer_dim_shared_net
        last_layer_dim_af_net = last_layer_dim_shared_net
        # value_only_layers maybe null list
        if len(vf_only_layers) > 0:
            last_layer_dim_vf_net = last_layer_dim_shared_net  # value net input dim
            for vf_layer_dim in vf_only_layers:
                assert isinstance(vf_layer_dim, int), "Error: net_arch[-1]['vf'] must only contain integers."
                vf_net.append(nn.Linear(last_layer_dim_vf_net, vf_layer_dim))  # add Linear layer
                vf_net.append(activation_fn())  # add activation function
                last_layer_dim_vf_net = vf_layer_dim

        if len(af_only_layers) > 0:
            last_layer_dim_af_net = last_layer_dim_shared_net  # policy net input dim
            for pi_layer_dim in af_only_layers:
                assert isinstance(pi_layer_dim, int), "Error: net_arch[-1]['af'] must only contain integers."
                af_net.append(nn.Linear(last_layer_dim_af_net, pi_layer_dim))
                af_net.append(activation_fn())
                last_layer_dim_af_net = pi_layer_dim

        self.latent_vf_dim = last_layer_dim_vf_net
        self.latent_af_dim = last_layer_dim_af_net

        self.shared_net = nn.Sequential(*shared_net)
        self.vf_net = nn.Sequential(*vf_net)
        self.af_net = nn.Sequential(*af_net)



"""
DualingFore类的net_arch参数中的layer都表示隐藏层。
DualingFore构造的网络接收特征，输出特征。具体的价值或动作的生成需要在Actor和Critic或Policy中具体写出。
DualingFore的share net和分支net可以是空，表示不存在对应的处理过程，输入直接为输出。

参数格式： net_arch = [{'vf':[], 'qf':[], 128, 128}]

DualingFore_StateAction
    ***********            ************
     State Net              Action Net
    ***********            ************
         |                      |
     latent_vf              latent_qf
         |______________________| 
                    |
          **********************
             Actor Shared Net
          **********************   
                    |
                 q value                  



"""


class DualingFore_StateAction(nn.Module):
    '''
    vf 表示处理状态的一支
    qf 表示处理动作的一支
    '''
    def __init__(self,
                 state_dim,
                 action_dim,
                 net_arch: List[Union[int, Dict[str,List[int]]]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 dropout: float = 0,
                 ):
        super(DualingFore_StateAction, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_arch = net_arch
        self.dropout = dropout

        # in case of the net arch is None
        shared_net = []
        vf_only_layers = []
        qf_only_layers = []
        last_layer_dim_vf_net = state_dim
        last_layer_dim_qf_net = action_dim
        self.last_layer_dim_shared_net = state_dim + action_dim  # in case state_only_layers and action_only_layers is None

        for layer in net_arch:
            # From here on the network splits up in policy and value network
            if isinstance(layer, dict):
                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[0]['vf'] must contain a list of integers."
                    vf_only_layers = layer["vf"]
                    if len(layer['vf']) > 0:
                        last_layer_dim_vf_net = layer["vf"][-1]

                if "qf" in layer:
                    assert isinstance(layer["qf"], list), "Error: net_arch[1]['qf'] must contain a list of integers."
                    qf_only_layers = layer["qf"]
                    if len(layer['qf']) > 0:
                        last_layer_dim_qf_net = layer["qf"][-1]
                self.last_layer_dim_shared_net = last_layer_dim_vf_net + last_layer_dim_qf_net
            else:
                assert isinstance(layer, int), "Error: the net_arch list can only contain ints and dicts"
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(self.last_layer_dim_shared_net, layer))  # add linear of size layer
                shared_net.append(activation_fn())
                if self.dropout > 0:
                    shared_net.append(nn.Dropout(self.dropout))
                self.last_layer_dim_shared_net = layer

        # get vf_net from vf_only_layers
        vf_net = []
        if len(vf_only_layers) > 0:
            last_layer_dim = state_dim
            for layer in vf_only_layers:
                assert isinstance(layer, int), "Error: net_arch[0]['vf'] must only contain integers."
                vf_net.append(nn.Linear(last_layer_dim, layer))  # add Linear layer
                vf_net.append(activation_fn())
                last_layer_dim = layer

        # get qf_net from qf_only_layers
        qf_net = []
        if len(qf_only_layers) > 0:
            last_layer_dim = action_dim
            for layer in qf_only_layers:
                assert isinstance(layer, int), "Error: net_arch[0]['qf'] must only contain integers."
                qf_net.append(nn.Linear(last_layer_dim, layer))  # add Linear layer
                qf_net.append(activation_fn())
                last_layer_dim = layer

        self.latent_dim = self.last_layer_dim_shared_net
        self.vf_net = nn.Sequential(*vf_net)
        self.qf_net = nn.Sequential(*qf_net)
        self.shared_net = nn.Sequential(*shared_net)

    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor
                ):

        latent_vf = self.vf_net(states)
        latent_qf = self.qf_net(actions)
        shared_latent = torch.cat((latent_vf, latent_qf), dim=1)
        return self.shared_net(shared_latent)

def create_cnn(
        input_channel: int,
        num_of_filters: List[int],
        kernels_size: List[int],
        strides: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
):
    # without pooling, BN..., to be continued...

    def cnn_block(num_in_filters: int, num_out_filters: int, kernel_size: tuple, stride: tuple) -> List[nn.Module]:
        # add a conv layer and an activation function

        layers = [nn.Conv2d(in_channels=num_in_filters,
                            out_channels=num_out_filters,
                            kernel_size=kernel_size,
                            stride=stride),
                  activation_fn()]
        return layers

    num_of_filters.insert(0, input_channel)   # note: num_of_filters will change in-place
    cnn_modules = []
    for i in range(len(num_of_filters) - 1):
        cnn_modules.extend(cnn_block(num_of_filters[i], num_of_filters[i + 1], kernels_size[i], strides[i]))
    cnn = nn.Sequential(*cnn_modules)
    return cnn

# cnn = create_cnn(input_channel=4, num_of_filters=[32,64,64], kernels_size=[8,4,3], strides=[4,2,1])
# print(cnn)

# net_arch = []
# net_arch = [128, 128, {'vf':[64,64], 'af':[128,128]}]
# net_arch = [{'pi':[64,64],'vf':[32,32]}]
# net_arch = [128,128,{'vf':[32,32]}]
# net_arch = [128,128,{'vf':[]}]

# net_arch = [{'vf':[128,64],'qf':[64,32]},64,32]
# net_arch = [{'vf':[128,64],'qf':[64,32]}]  # 退化
# net_arch = [{'vf':[128,64]},64,32]  # action直接送入shared net
# mlp = DualingPost_ValueAdvantage(
#     net_arch=net_arch,
#     input_dim = 291,
#     activation_fn=nn.Tanh,
# )
# print(mlp)




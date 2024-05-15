import torch
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable


def get_optimizer(optim_aliase:str, parameters, lr:float) -> Type[torch.optim.Optimizer]:

    optim_aliases: Dict[str, Type[torch.optim.Optimizer]] = {"sgd": torch.optim.SGD,
                                                             "adam": torch.optim.Adam,
                                                             "rmsprop": torch.optim.RMSprop, }

    if optim_aliase is None:
        optim_aliase = 'adam'
        optim = torch.optim.Adam(parameters,lr)
    elif optim_aliase in optim_aliases:
        optim_ = optim_aliases[optim_aliase]
        optim = optim_(parameters,lr)
    else:
        raise ValueError(f'optim_aliase : {optim_aliase} unknown')
    print(f'optim:{optim_aliase} !')
    return optim
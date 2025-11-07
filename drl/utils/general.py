from typing import Union, Dict, Tuple, Any, TypeAlias

import torch
import logging
import numpy as np
from pathlib import Path
from torch_cluster import knn
from tabulate import tabulate
from collections import defaultdict
from time import gmtime, strftime, struct_time
from torch.optim.optimizer import ParamsT


###############################################################################################
############################################## TYPING #########################################
###############################################################################################
Device: TypeAlias = Union[None, str, torch.device]
Params: TypeAlias = Union[ParamsT, Dict[str, ParamsT]]
Buffer: TypeAlias = Union[np.ndarray, torch.Tensor, Tuple[np.ndarray, ...], Tuple[torch.Tensor, ...], Dict[str, Any]]

###############################################################################################
################################### LOGGING AND FORMAT ########################################
###############################################################################################
LOGGER = logging.getLogger(name='DRL')
LOGGER.setLevel(logging.INFO)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(logging.Formatter('%(message)s'))
LOGGER.addHandler(streamHandler)


def format_time(time_delta: float) -> str:
    seq_time = list(gmtime(time_delta))
    if 0 <= time_delta < 60:
        template = '%Ss'
    elif 60 <= time_delta < 3600:
        template = '%Mm %Ss'
    elif 3600 <= time_delta < 86400:
        template = '%Hh %Mm %Ss'
    else:
        template = '%dd %Hh %Mm %Ss'
        seq_time[2] -= 1
    seq_time = struct_time(seq_time)
    return strftime(template, seq_time)


def format_tabulate(
    title: str,
    subtitle: str,
    results: Dict[str, Any],
    tail_info: str=''
) -> str:
    content = ''
    content += f'*** {title}:[{subtitle}]\n'
    content += tabulate(tabular_data=[results.values()], headers=results.keys(), tablefmt='orgtbl')
    content += f'\n--- {tail_info}'
    return content


def increase_exp(
    project_path: Union[str, Path] = '',
    name: str = 'exp'
) -> Path:
    if project_path == '':
        project_path = 'runs'
    project_path = Path(project_path)
    if not project_path.exists():
        project_path.mkdir(parents=True)

    # create experiment directory with exp_id has not existed
    exp_id = 0
    while (project_path / f'{name}{exp_id}').exists():
        exp_id += 1
    (project_path / f'{name}{exp_id}').mkdir(parents=True, exist_ok=True)
    return project_path / f'{name}{exp_id}'
    


###############################################################################################
###################################### CALCULATIONS ###########################################
###############################################################################################
def random_mixup_data(
    policy_data: Dict[str, Any], 
    expert_data: Dict[str, Any], 
    requires_grad: bool = False
) -> Dict[str, Any]:
    if isinstance(policy_data, dict):
        assert isinstance(expert_data, dict), TypeError
        return {name: random_mixup_data(policy_data[name], expert_data[name]) for name in policy_data}
    elif isinstance(policy_data, (list, tuple)):
        assert isinstance(expert_data, (list, tuple)), TypeError
        return [random_mixup_data(pdata, edata) for pdata, edata in zip(policy_data, expert_data)]
    else:
        r_mix = torch.rand(policy_data.size(0), *((policy_data.ndim - 1) * [1]), device=policy_data.device)
        mix_data = r_mix * policy_data + (1 - r_mix) * expert_data
        mix_data.requires_grad = requires_grad
        return mix_data


def query_knn_idxs(
    s: torch.Tensor, D: torch.Tensor, 
    k: int, cosine: bool = False
) -> torch.Tensor:
    assert (s.ndim == D.ndim == 2) and (s.size(-1) == D.size(-1)), \
        ValueError(f'Invalid s ({s.size()}) and D shapes ({D.size()})')
    assert (D.size(0) >= k), ValueError(f'Invalid number of target points ({D.size(0)}).')
    assign_indexes = knn(x=D, y=s, k=k, cosine=cosine)[1]
    return assign_indexes.reshape(len(s), k)


def query_knn_radius(
    s: torch.Tensor, D: torch.Tensor, k: int
) -> torch.Tensor:
    knn_idxs = query_knn_idxs(s=s, D=D, k=k) # (Nq, k)
    return torch.linalg.norm(s - D[knn_idxs[:, -1]], dim=-1)


class MeanMetrics(defaultdict):
    def __init__(self):
        super(MeanMetrics, self).__init__(lambda: 0)
        self.t = 1

    def update(self, metrics: Dict[str, Union[int, float]]):
        alpha = 1 / self.t
        for name, value in metrics.items():
            self[name] = (1 - alpha) * self[name] + alpha * value
        self.t += 1
    
    def reset(self):
        self.clear()
        self.t = 1


###############################################################################################
################################# STRUCTURE MANIPULATION ######################################
###############################################################################################
from tree import map_structure, flatten as flatten_structure


def put_structure(struct_trg, index: int, struct_src):
    def put(x, y): 
        x[index] = y
    map_structure(put, struct_trg, struct_src)   


def groupby_structure(list_structs, func: callable = np.stack):
    if isinstance(list_structs[0], (list, tuple)):
        list_structs = list(zip(*list_structs))
        return [groupby_structure(child_struct, func) for child_struct in list_structs]
    return map_structure(lambda *args: func(args), *list_structs)


def nearest_node_value(func: callable, struct):
    if isinstance(struct, (list, tuple, dict)):
        for child_struct in (struct.values() if isinstance(struct, dict) else struct):
            return nearest_node_value(func, child_struct)
    return func(struct)
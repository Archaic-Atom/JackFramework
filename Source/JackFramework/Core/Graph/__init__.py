# -*- coding: UTF-8 -*-
from JackFramework.SysBasic.switch import Switch
from JackFramework.SysBasic.log_handler import LogHandler as log

from .build_training_graph import BuildTrainingGraph
from .build_testing_graph import BuildTestingGraph
from .data_handler_manager import DataHandlerManager


def _get_graph_dict() -> dict:
    return {
        'train': BuildTrainingGraph,
        'test': BuildTestingGraph,
        'background': BuildTestingGraph,
        'online': None,
        'reinforcement_learning': None,
        'web': BuildTestingGraph
    }


def graph_selection(args: object, jf_model: object) -> object:
    graph_dict = _get_graph_dict()
    assert args.mode in graph_dict
    log.info(f'Enter {args.mode} mode')
    return graph_dict[args.mode](args, jf_model)


def dataloader_selection(args: object, jf_dataloader: object) -> object:
    return DataHandlerManager(args, jf_dataloader)

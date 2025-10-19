# -*- coding: UTF-8 -*-
"""Graph selection utilities used by the different modes."""

from typing import Dict, Type

from JackFramework.SysBasic.log_handler import LogHandler as log

from .build_testing_graph import BuildTestingGraph
from .build_training_graph import BuildTrainingGraph
from .data_handler_manager import DataHandlerManager


GRAPH_REGISTRY: Dict[str, Type] = {
    'train': BuildTrainingGraph,
    'test': BuildTestingGraph,
    'background': BuildTestingGraph,
    'web': BuildTestingGraph,
}


def graph_selection(args: object, jf_model: object) -> object:
    try:
        graph_cls = GRAPH_REGISTRY[args.mode]
    except KeyError as exc:
        available = ', '.join(sorted(GRAPH_REGISTRY))
        raise ValueError(f'Unsupported graph mode `{args.mode}`. Available: {available}.') from exc

    log.info(f'Constructing graph for `{args.mode}` mode')
    return graph_cls(args, jf_model)


def dataloader_selection(args: object, jf_dataloader: object) -> object:
    return DataHandlerManager(args, jf_dataloader)

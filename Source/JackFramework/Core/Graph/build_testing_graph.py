# -*- coding: utf-8 -*-
import os
import torch

from .meta_ops import MetaOps


class BuildTestingGraph(MetaOps):
    def __init__(self, args: object, jf_model: object, rank: object) -> object:
        super().__init__(args, jf_model, rank)
        self.__args = args

    def exec(self, input_data: list, label_data: list = None, is_training: bool = False) -> list:
        assert label_data is None and not is_training
        input_data = self._pass_data2device(input_data)
        outputs_data = []
        with torch.no_grad():
            for i, model_item in enumerate(self._model):
                output_data = self.inference(model_item, input_data, i)
                outputs_data.append(output_data)

        return outputs_data

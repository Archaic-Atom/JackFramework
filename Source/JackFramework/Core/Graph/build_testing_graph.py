# -*- coding: utf-8 -*-
import torch
from ._meta_ops import MetaOps


class BuildTestingGraph(MetaOps):
    def __init__(self, args: object, jf_model: object) -> None:
        super().__init__(args, jf_model)

    def exec(self, input_data: list, label_data: list = None, is_training: bool = False) -> list:
        if label_data is not None or is_training:
            raise ValueError('Testing graph expects no labels and runs in eval mode only.')
        if not self._model:
            raise RuntimeError('Model has not been initialised.')
        input_data = self._pass_data2device(input_data)
        outputs_data = []
        with torch.no_grad():
            for i, model_item in enumerate(self._model):
                output_data = self.user_inference(model_item, input_data, i)
                outputs_data.append(output_data)
        return outputs_data

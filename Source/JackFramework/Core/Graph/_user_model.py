# -*- coding: utf-8 -*-
from JackFramework.SysBasic.log_handler import LogHandler as log
from JackFramework.UserTemplate import ModelHandlerTemplate
from JackFramework.SysBasic.show_handler import ShowHandler


class UserModel(ShowHandler):
    def __init__(self, args: object, jf_model: ModelHandlerTemplate) -> None:
        super().__init__()
        if not isinstance(jf_model, ModelHandlerTemplate):
            raise TypeError('jf_model must implement ModelHandlerTemplate.')
        self.__jf_model = jf_model
        self.__args = args
        self.__model, self.__opt, self.__sch = None, None, None

    @property
    def _model(self):
        return self.__model

    @property
    def _opt(self):
        return self.__opt

    @property
    def _sch(self):
        return self.__sch

    def user_pretreatment(self, epoch: int) -> None:
        self.__jf_model.pretreatment(epoch, self.rank)

    def user_post_process(self, epoch: int, ave_tower_loss: list = None,
                          ave_tower_acc: list = None) -> None:
        self.__jf_model.post_process(epoch, self.rank, ave_tower_loss, ave_tower_acc)

    def user_inference(self, model_item: object, input_data: list, model_id: int) -> list:
        return self.__jf_model.inference(model_item, input_data, model_id)

    def user_loss(self, output_data: list, label_data: list, model_id: int) -> list:
        return self.__jf_model.loss(output_data, label_data, model_id)

    def user_accuracy(self, output_data: list, label_data: list, model_id: int) -> list:
        return self.__jf_model.accuracy(output_data, label_data, model_id)

    def user_load_model(self, checkpoint: dict, model_id: int) -> bool:
        return self.__jf_model.load_model(self._model[model_id], checkpoint, model_id)

    def user_load_opt(self, checkpoint: dict, opt_id: int) -> bool:
        return self.__jf_model.load_opt(self._opt[opt_id], checkpoint, opt_id)

    def user_save_model(self, epoch: int) -> dict:
        return self.__jf_model.save_model(epoch, self.__model, self.__opt)

    def user_lr_scheduler(self, sch_item: object, loss: list, sch_id: int) -> None:
        self.__jf_model.lr_scheduler(sch_item, float(loss[sch_id][0]), sch_id)

    def user_init_model(self) -> None:
        log.info("Loading user's model!")
        self.__model = self.__jf_model.get_model()
        log.info("Successfully get user's model!")

    def user_init_optimizer(self) -> None:
        log.info("Loading user's optimizer!")
        self.__opt, self.__sch = self.__jf_model.optimizer(self.__model, self.__args.lr)
        log.info("Successfully get user's optimizer!")

    # Framework-side release to allow DDP modules to be GC'ed before PG destroy
    def free_model(self) -> None:
        self.__model = None
        self.__opt = None
        self.__sch = None

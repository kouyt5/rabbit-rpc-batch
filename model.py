from abc import ABCMeta, abstractmethod
import logging
import time
import random


class DispatchModel(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass
    

    # @abstractmethod
    def __call__(self, input_lists:list):
        """
        """
        logging.info("compute a element")
        time.sleep(0.2)
        return input_lists

    @abstractmethod
    def schedulable_infer(self, *args, **kwargs):
        """
        实现可调度模型推理
        """
        raise NotImplementedError("schedulable_infer方法未实现")

class ImplDispatchModel(DispatchModel):
    def __init__(self) -> None:
        super().__init__()

    def schedulable_infer(self, input_lists:list):
        logging.info("compute a element")
        time.sleep(float(random.randint(200, 400))/1000)
        return input_lists


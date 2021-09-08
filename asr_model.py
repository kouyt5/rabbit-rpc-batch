

from model import DispatchModel
import random
import logging
import time


class AsrModel(DispatchModel):
    def __init__(self) -> None:
        super().__init__()

    def schedulable_infer(self, input_lists:list):
        logging.info("compute asr batch")
        time.sleep(float(random.randint(200, 400))/1000)
        return [str(len(input_lists[i].read())) for i in range(len(input_lists))]  # fake data
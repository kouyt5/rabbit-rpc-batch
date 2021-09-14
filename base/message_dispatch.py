

from abc import ABCMeta, abstractmethod
from typing import List
from base.service import Service

import pika


class MessageDispatch(metaclass=ABCMeta):

    def __init__(self, callback) -> None:
        """
        :params
            callback: 回调函数，用于消息后处理
        """
        self.callback = callback

    @abstractmethod
    def deliver_message(self, body:bytes, properties:pika.BasicProperties, delivery_tag:str) -> None:
        """
        接受消息处理层的消息
        """
        pass

class SimpleMessageDispatch(MessageDispatch):
    """
    简单的消息分配器实现
    """

    def __init__(self, callback, service:Service) -> None:
        super().__init__(callback)
        self.service = service

    def deliver_message(self, body: bytes, properties: pika.BasicProperties, delivery_tag: str) -> None:
        # 处理消息
        result = self.service(bodys=body)
        self.callback(result, properties.correlation_id, properties.reply_to, delivery_tag)

class SimpleService(Service):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, bodys: List[bytes]) -> List[bytes]:
        return bodys

if __name__ == '__main__':
    service = SimpleService()
    message_dispatch = SimpleMessageDispatch(None, service=service)
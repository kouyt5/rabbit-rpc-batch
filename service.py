from abc import ABCMeta
from typing import List


class Service(metaclass=ABCMeta):

    def __init__(self) -> None:
        pass

    def __call__(self, bodys:List[bytes]) -> List[bytes]:
        raise NotImplementedError("算法模块未实现")

    def pack_msg(self, one_body:bytes) -> bytes:
        """
        封装一个消息，用特定格式
        """
        return one_body
    
    def unpack_msg(self, one_body:bytes) -> bytes:
        """
        接封装一个消息
        """
        return one_body
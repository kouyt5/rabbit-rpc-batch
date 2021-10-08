from pika.exceptions import AMQPConnectionError
import functools
from rpcb.log import logger
from time import sleep
import sys


def cycle_connect(sleep_time=3.0):
    """
    Params:
        sleep_time: 睡眠时间
        
    rabbitmq连接失效后程序将会退出，因此使用该decorator
    用来检测异常然后反复连接。考虑到很多模块将会使用该功能
    因此使用decorator是个不错的选择，可以避免大量的try catch出现
    在程序中了

    Examples:
    >>> from common.decorator import cycle_connect
    >>> @cycle_connect(5)
    >>> def test():
    >>>     print("hello world)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            while(True):
                try:
                    func(*args, **kwargs)
                except AMQPConnectionError as e:
                    logger.error(e.args)
                    sleep(sleep_time)  # sleep_time秒后重新连接
                    logger.warning("尝试重新连接")
                except KeyboardInterrupt:
                    logger.warning("程序被 Crtl+C 强制退出....")
                    sys.exit()
        return wrapper
    return decorator

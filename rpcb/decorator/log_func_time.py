import functools
from rpcb.log import logger
import time


def log_func_time(message='', require_return=True):
    """
    Parms：
        message: 描述函数功能的字符串
        require_return: 函数是否需要返回

    统计函数运行时间，在函数末尾打印运行时间

    Example:
    >>> from common.decorator import log_fun_time
    >>> @log_fun_time('测试函数运行')
    >>> def test():
    >>>     pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if require_return:
                start_time = time.time()
                result = func(*args, **kwargs)
                logger.info(message + '用时：' +
                            str(time.time()-start_time) + 's')
                return result
            else:
                start_time = time.time()
                func(*args, **kwargs)
                logger.info(message + '用时：' +
                            str(time.time()-start_time) + 's')
        return wrapper
    return decorator

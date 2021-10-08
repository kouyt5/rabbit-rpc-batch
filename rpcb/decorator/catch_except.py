import functools

import msgpack
from rpcb.log import logger
from rpcb.exception.audio_exception import AudioFormatUnsupportedException
from rpcb.status import get_response_by_status
from rpcb.status.status_util import Status


def catch_except(func):
    """
    抓取异常，封装传给java端
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError as e:
            logger.error(e.args)
            logger.warning("请检查传入参数类型")
            return get_response_by_status(Status.DataFormatError, None)
        except AudioFormatUnsupportedException as e:
            logger.error("音频格式不正确", exc_info=True)
            return get_response_by_status(Status.AudioFormatUnsupportedError, None)
        except RuntimeError as e:
            logger.error(e.args, exc_info=True)
            return get_response_by_status(Status.RuntimeError, None)
        except Exception as e:
            logger.error(e.args, exc_info=True)
            logger.warning("未知异常")
            return get_response_by_status(Status.ServiceUnknownException, None)
    return wrapper

def catch_excepts(func):
    """
    抓取多个异常，传入一个list对象，封装传给java端
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError as e:
            logger.error(e.args)
            logger.warning("请检查传入参数类型")
            return [msgpack.packb(get_response_by_status(Status.DataFormatError, None))
                         for i in range(len(args))]
        except RuntimeError as e:
            logger.error(e.args)
            return [msgpack.packb(get_response_by_status(Status.RuntimeError, None)) 
                        for i in range(len(args))]
        except Exception as e:
            logger.error(e.args)
            logger.warning("未知异常")
            return [msgpack.packb(get_response_by_status(Status.ServiceUnknownException, None)) 
                        for i in range(len(args))]
    return wrapper
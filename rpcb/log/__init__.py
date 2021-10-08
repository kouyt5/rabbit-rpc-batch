import logging
from .base_log import Logger


# _log_dir = get_conf()['log']['log_dir']  # 默认在启动的代码的当前目录下新建一个log目录
# _level = get_conf()['log']['log_level']
_log_dir = 'logs'
_level = logging.DEBUG
# 外部模块使用from base_log import logger 获取logger对象直接使用
logger = Logger(__name__, _level, _log_dir).get_logger()

__all__ = ["Logger", "logger"]

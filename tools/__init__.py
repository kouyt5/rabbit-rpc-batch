import logging
from .logger import Logger


_log_dir = 'logs'  # 默认在启动的代码的当前目录下新建一个log目录
_level = logging.INFO
# 外部模块使用from base_log import logger 获取logger对象直接使用
logger = Logger(__name__, _level, _log_dir, 10*1024*1024, 7).get_logger()

__all__ = ["Logger", "logger"]
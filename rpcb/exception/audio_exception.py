# coding=utf-8

class ServerRabbitMqTimeoutException(Exception):
    """
    多元模块调用超时
    """
    def __init__(self):
        pass

class AudioFormatUnsupportedException(Exception):
    """
    音频格式不支持
    """
    def __init__(self):
        pass

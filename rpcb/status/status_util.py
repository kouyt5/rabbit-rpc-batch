

from enum import Enum


class Status(Enum):
    Success = 200
    ServiceUnknownException = 400
    RuntimeError = 401
    AudioFormatUnsupportedError = 402
    DataFormatError = 403
    ServiceRabbitMqMessageTimeOut = 404
    ServiceRabbitMqMessagePublishFailed = 405
    ServiceRabbitMessageIsNull = 408
    ServiceMessagePackSerializationFailed = 406
    ServiceMessagePackDeserializationFailed = 407
    ServiceAlgorithmException = 409
    ServiceRPCFailed = 410


_STATUS_DESC_DICT = {
    200: "成功",
    400: "服务端未定义异常",
    401: "运行状态错误",
    402: "音频格式不支持",
    403: "服务端接收到的数据无法解析",
    404: "服务端接收消息超时",
    405: "服务端发布消息失败",
    408: "服务端接收到的消息为空",
    406: "服务端消息序列化失败",
    407: "服务端消息反序列化失败",
    409: "服务端算法处理异常",
    410: "服务端调用中断"
}


def get_response_by_status(status:Status, result: any) -> dict:
    """
    Params:
        status: 枚举实体
        result: 处理的结果
        
    返回给java封装的数据
    """
    
    if status.name not in Status._member_names_:
        raise status + "状态未定义"
    status = status.value  # 状态码
    desc = _STATUS_DESC_DICT[status]  # 状态描述
    return {
        'status': status,
        'message': desc,
        'data': result
    }


def get_success_response(result: any) -> dict:
    """
    Params:
        result: 处理的结果

    返回给java封装的处理成功的数据,
    """
    return get_response_by_status(Status.Success, result)
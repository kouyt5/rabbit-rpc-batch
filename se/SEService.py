import sys, os
sys.path.append(os.path.abspath('.'))
from rpcb.decorator.log_func_time import log_func_time
from rpcb.decorator.catch_except import catch_excepts

from se.SEModel import SEModel
from rpcb.model import DispatchModel
from rpcb.service import Service
import os
from typing import Any, List, Tuple
import time, io
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import msgpack
from concurrent.futures import ThreadPoolExecutor
from rpcb.status.status_util import Status, get_response_by_status
from rpcb.log import logger


class SEService(Service):
    def __init__(self, model: DispatchModel) -> None:
        super().__init__()
        self.model = model
        self.pool = ThreadPoolExecutor(max_workers=os.cpu_count())

    @log_func_time(require_return=True, message="语音增强耗时：")
    @catch_excepts
    def __call__(self, bodys: List[bytes]) -> List[bytes]:
        unpacked_result = [self.unpack_msg(body) for body in bodys]  # unpack后的数据[{key, value}...]
        model_needed_datas, status = self.prepare_data(unpacked_result)  # 处理音频为模型支持的输入格式
        # 算法结果
        io_bytes_lists, durations = self.model.schedulable_infer(model_needed_datas)  # ["你好", "今天天气..",....]
        model_results = (io_bytes_lists, durations, status)
        # 打包
        return [self.pack_msg(*model_result) for model_result in zip(*model_results)]

    def pack_msg(self, enh_audio_io:io.BytesIO, duration:float, status:Status) -> Any:
        """
        封装一个消息，用特定格式
        """
        body = {
            'enhance': enh_audio_io.getvalue(),
            'duration': "{:0.2f}".format(duration)
        }
        return msgpack.packb(get_response_by_status(status=status, result=body))

    def unpack_msg(self, one_body: bytes) -> dict:
        """
        接封装一个消息
        """
        return msgpack.unpackb(one_body)

    def prepare_data(self, datas: List[dict]) -> Tuple[List[io.BytesIO], List[str]]:
        """
        如果有必要，对音频做处理
        :params
            datas: 原始的数据列表[{key,value...},...]
        :returns
            文件数据列表
        """
        # 列表转字典
        datas_seq = [(i, datas[i]) for i in range(len(datas))]  # 把列表转换为有序[(0, xx), (1, xx)...]
        # 考虑到音频格式转换比较费时，这里使用多线程处理
        pre_time = time.time()
        results = self.pool.map(self.transform2wav, datas_seq)
        # 打包结果转换
        results = list(results)
        results.sort(key=lambda item: item[0])
        seq_result = [results[i][1] for i in range(len(results))]  # [io.BytesIO(), io.BytesIO()....]
        logger.info("音频格式转换用时:" + str(time.time() - pre_time))
        return seq_result, [results[i][2] for i in range(len(results))]  # [io.BytesIO(),...], [200,200...]

    # try catch如果异常如何返回
    def transform2wav(self, data_seq):
        seq = data_seq[0]  # 顺序编号
        data = data_seq[1]  # 数据
        # TODO 如果格式不对？
        audio = data['audio']  # bytes
        audio = io.BytesIO(audio)
        format = data['format']
        out_io = io.BytesIO()
        status = Status.Success
        if format == 'aac' or format == 'mp3':
            try:
                data = AudioSegment.from_file(audio)
                data.export(out_f=out_io, format='wav')
            except:
                logger.error("音频格式解析错误")
                sf.write(out_io, np.zeros((1600,), dtype=np.float32), samplerate=16000,
                         subtype='PCM_16', format='WAV')
                out_io.seek(0)
                status = Status.DataFormatError
        elif format == 'wav':
            out_io = audio  # 如果wav格式有问题，由下一层处理
        else:
            logger.error("音频格式不支持, 返回0 zero padding...")
            sf.write(out_io, np.zeros((1600,), dtype=np.float32), samplerate=16000,
                     subtype='PCM_16', format='WAV')
            out_io.seek(0)
            status = Status.AudioFormatUnsupportedError
        return (seq, out_io, status)

if __name__ == '__main__':
    model_path = "/data/chenc/asr/rabbit-rpc-batch/se/checkpoints/dns48/pretrain.th"
    test_audio = ["/data/chenc/asr/rabbit-rpc-batch/wav/0.wav",
                "/data/chenc/asr/rabbit-rpc-batch/wav/7.wav"]
    se_model = SEModel(model_path=model_path,
                        device="cpu", dry=0)
    service = SEService(se_model)

    data_list = []
    for i in range(len(test_audio)):
        data_list.append({
            "audio": io.FileIO(test_audio[i]).read(),
            "format": 'wav'
        })

    data_list = [msgpack.packb(item) for item in data_list]
    result = service(data_list)
    print(result)
from base.model import DispatchModel, ImplDispatchModel
from base.service import Service
import os
from typing import List
import multiprocessing as mp
import time, io

import numpy as np
from pydub import AudioSegment
import logging
import soundfile as sf
from functools import partial
import msgpack
from concurrent.futures import ThreadPoolExecutor


class AsrService(Service):
    def __init__(self, model:DispatchModel) -> None:
        super().__init__()
        self.model = model
        self.pool = ThreadPoolExecutor(max_workers=os.cpu_count())

    
    def __call__(self, bodys: List[bytes]) -> List[bytes]:
        unpacked_result = [self.unpack_msg(body) for body in bodys]  # unpack后的数据[{key, value}...]
        model_needed_datas = self.prepare_data(unpacked_result)  # 处理音频为模型支持的输入格式
        # 算法结果
        model_results = self.model.schedulable_infer(model_needed_datas)  # ["你好", "今天天气..",....]
        # 结果封装
        total_result = []
        for model_result in model_results:
            total_result.append({
                "sentence": model_result,
                "other": "other"
            })

        return [self.pack_msg(result) for result in total_result]

    def pack_msg(self, one_body:bytes) -> bytes:
        """
        封装一个消息，用特定格式
        """
        return msgpack.packb(one_body)
    
    def unpack_msg(self, one_body:bytes) -> bytes:
        """
        接封装一个消息
        """
        return msgpack.unpackb(one_body)

    def prepare_data(self, datas:List[dict]) -> List[io.BytesIO]:
        """
        如果有必要，对音频做处理
        :params
            datas: 原始的数据列表[{key,value...},...]
        :returns
            文件数据列表
        """
        # 列表转字典
        datas_seq = [(i, datas[i]) for i in range(len(datas))] # 把列表转换为有序[(0, xx), (1, xx)...]
        # 考虑到音频格式转换比较费时，这里使用多线程处理
        pre_time = time.time()
        results = self.pool.map(self.transform2wav, datas_seq)
        # 打包结果转换
        results = list(results)
        results.sort(key=lambda item:item[0])
        seq_result = [results[i][1] for i in range(len(results))]  # [io.BytesIO(), io.BytesIO()....]
        logging.info("线程运行+结果封装:"+str(time.time()-pre_time))
        return seq_result
        

    # try catch如果异常如何返回
    def transform2wav(self, data_seq):
        seq = data_seq[0]  # 顺序编号
        data = data_seq[1]  # 数据
        # TODO 如果格式不对？
        audio = data['audio']  # bytes
        audio = io.BytesIO(audio)
        format=data['format']
        out_io = io.BytesIO()
        if format=='aac' or format=='mp3':
            try:
                data = AudioSegment.from_file(audio)
                data.export(out_f=out_io, format='wav')
            except:
                logging.error("音频格式解析错误")
                sf.write(out_io, np.zeros((1600,), dtype=np.float32), samplerate=16000, 
                                subtype='PCM_16', format='WAV')
                out_io.seek(0)
        elif format=='wav':
            out_io = audio  # 如果wav格式有问题，由下一层处理
        else:
            logging.error("音频格式不支持, 返回0 zero padding...")
            sf.write(out_io, np.zeros((1600,), dtype=np.float32), samplerate=16000, 
                            subtype='PCM_16', format='WAV')
            out_io.seek(0)
        return (seq, out_io)
        

        
if __name__ == '__main__':

    service = AsrService(ImplDispatchModel())
    data_list = []
    file_list = os.listdir('wav')
    for i in range(len(file_list)):
        data_list.append({
            "audio": io.FileIO(os.path.join('/home/chenc/workspace/rabbit-py/wav',file_list[i])).read(),
            "format": 'wav'
        })
    
    data_list = [msgpack.packb(item) for item in data_list]
    result = service(data_list)
    print(result)


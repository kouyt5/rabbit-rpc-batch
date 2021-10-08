import io
import logging
import math
import time
from typing import List
import soundfile as sf
import scipy.signal
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
import torchaudio
import torch


class SEAudioDataset:
    def __init__(self, sample_rate=16000):
        """
        音频数据集类
        """
        self.files = []
        self.sample_rate = sample_rate
        

    def set_files(self, files:List[str]):
        """
        给数据集合注入数据
        """
        self.files.extend(files)

    def clear_data(self):
        self.files.clear()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        data, sr = sf.read(file=file)
        pre_time = time.time()
        out_io = io.BytesIO()
        if not sr == self.sample_rate:
            data = scipy.signal.resample(data, int(self.sample_rate*len(data)/sr))
        sf.write(out_io, data, samplerate=self.sample_rate,subtype="PCM_16", format="WAV")
        out_io.seek(0)
        logging.debug("音频写入io对象花费: {:.3f}".format(time.time()-pre_time))
        out, sr = torchaudio.load(out_io)  # 1*16000
        return out

    def collate_fn(self, batchs):
        longest_sample_size = max(batchs, key=lambda x: x.size(1)).size(1)
        inputs_signals = torch.zeros(len(batchs), 1, longest_sample_size)
        inputs_lengths = torch.zeros(len(batchs))
        for i in range(len(batchs)):
            inputs_signals[i].narrow(1, 0, batchs[i].size(1)).copy_(torch.as_tensor(batchs[i][0]).squeeze(0))
            inputs_lengths[i] = torch.as_tensor(batchs[i].size(1), dtype=torch.int32)
        return inputs_signals, inputs_lengths

if __name__ == "__main__":
    dataset = SEAudioDataset(sample_rate=16000)
    file = "/mnt/d/minhang/atc-service/se/wav/jiafang-second-verify-normal/20210917130418-ASR-1706.wav"
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn)
    dataset.set_files([file, file])
    for batch in dataloader:
        batch



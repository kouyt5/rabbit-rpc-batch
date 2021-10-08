import io
import logging
import os
import sys
import time

import numpy as np
from torch.utils import data

sys.path.append(os.path.abspath('.'))
from typing import List, Optional, Union

import torch
from torch.utils.data.dataloader import DataLoader
from denoiser.demucs import Demucs
import soundfile as sf
from se.data_layer import SEAudioDataset
from rpcb.model import DispatchModel


class ModelConfig:
    # ROOT = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/"
    # DNS_48_URL = ROOT + "dns48-11decc9d8e3f0998.th"       hidden=48
    # DNS_64_URL = ROOT + "dns64-a7761ff99a7d5bb6.th"       hidden=64
    # MASTER_64_URL = ROOT + "master64-8a5dfb4bb92753dd.th" hidden=64
    model_path = None
    dns64 = False
    master64 = False
    dns48 = False

class SEModel(DispatchModel):
    def __init__(self, model_path:str, device:str, dry:float = 0, 
                    max_batch_size:int = 12):
        super().__init__()
        args = ModelConfig()
        args.model_path = model_path
        self.dry = dry
        self.max_batch_size = max_batch_size
        if device == "cpu":
            self.device = torch.device(device)
        else:  # cuda:0 cuda:1等
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = Demucs(hidden=48)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.audio_datasets = SEAudioDataset(sample_rate=16000)
        self.data_loader = DataLoader(dataset=self.audio_datasets, batch_size=max_batch_size, 
                                        collate_fn=self.audio_datasets.collate_fn)

    def schedulable_infer(self, wav: Union[List[str], List[io.BytesIO]]) -> List[io.BytesIO]:
        """
        返回音频io对象和时长
        """
        estimates, lengths = self.infer_wavs(wav)
        # 把音频写入io对象中
        all_results = []
        durations = []
        for i in range(estimates.size(0)):
            out_io = io.BytesIO()
            data = estimates[i][0].cpu().detach().numpy()[0:int(lengths[i].item())]
            sf.write(out_io, data=data, samplerate=16000, subtype="PCM_16", format="WAV")
            out_io.seek(0)
            all_results.append(out_io)
            durations.append(lengths[i].item()/16000)
        return all_results, durations


    @torch.no_grad()
    def infer_wavs(self, wavs: Union[List[str], List[io.BytesIO]] = None):
        """
        对多个音频进行推理
        """
        self.audio_datasets.clear_data()
        self.audio_datasets.set_files(wavs)
        noises, lengths = next(iter(self.data_loader))
        noises = noises.to(self.device)
        estimates = self.model(noises)

        estimates = (1 - self.dry) * estimates + self.dry * noises
        return estimates, lengths


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    model_path = "/data/chenc/asr/rabbit-rpc-batch/se/checkpoints/dns48/pretrain.th"
    test_audio = ["/data/chenc/asr/rabbit-rpc-batch/wav/0.wav",
                "/data/chenc/asr/rabbit-rpc-batch/wav/7.wav"]
    se_model = SEModel(model_path=model_path,
                        device="cpu", dry=0)
    wav_byte_io = se_model.schedulable_infer(test_audio)
    for i in range(len(wav_byte_io)):
        io.FileIO("test{:s}.wav".format(str(i)), mode='w').write(wav_byte_io[i].read())

    


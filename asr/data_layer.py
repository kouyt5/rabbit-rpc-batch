
# from grammar import convert_wave2np
from torch.utils.data.dataloader import DataLoader
from nemo.core.classes import IterableDataset
from nemo.core.classes.dataset import Dataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import torch
import numpy as np


# simple data layer to pass audio signal
class AudioDataLayer(Dataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True
        self.datasets = []
        
    
    def __getitem__(self, index):
        return self.datasets[index]
        
    def set_signal(self, signal):
        signal = signal.astype(np.float32) #/32768.
        signal_shape = signal.size
        self.datasets.append((signal, signal_shape))
        self.output = True

    def clear(self):
        self.datasets.clear()

    def collate_fn(self, batchs):
        batchs
        longest_sample_size = max(batchs, key=lambda x: x[1])[1]
        inputs_signals = torch.zeros(len(batchs), longest_sample_size)
        inputs_lengths = torch.zeros(len(batchs))
        for i in range(len(batchs)):
            inputs_signals[i].narrow(0, 0, batchs[i][1]).copy_(torch.as_tensor(batchs[i][0]).squeeze(0))
            inputs_lengths[i] = torch.as_tensor(batchs[i][1], dtype=torch.int32)
        return inputs_signals, inputs_lengths


    def __len__(self):
        return len(self.datasets)

# if __name__ == '__main__':
#     dataset = AudioDataLayer(16000)
#     np_audio4, secs = convert_wave2np("4.wav")
#     np_audio1, secs = convert_wave2np("1.wav")
#     dataset.set_signal(np_audio1)
#     dataset.set_signal(np_audio4)
#     data_loader = DataLoader(dataset=dataset, batch_size=2,collate_fn=dataset.collate_fn)
#     for data in enumerate(data_loader):
#         data
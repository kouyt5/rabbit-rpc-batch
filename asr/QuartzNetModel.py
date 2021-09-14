from base.model import DispatchModel
import time
from typing import List, Tuple, Union
import nemo
from nemo.collections.asr.metrics.wer import WER
import torch
from asr.data_layer import AudioDataLayer
from torch.utils.data import DataLoader
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
import copy
import numpy as np
import soundfile as sf
import scipy.signal
import io
import logging
import psutil,os


class QuartznetModel(DispatchModel):
    """
    Quartznet15x5 语音识别模型封装类
    """
    def __init__(self, model_path, 
                        device:str="cpu", # alternative "cuda:0"
                        lang:str='cn', # 英文为'en'
                        beam_width:int = 20, # 语言模型搜索宽度
                        use_lm:bool = True, # 是否使用语言模型
                        lm_path:str = None, # 语言模型路径
                        alpha:float = 2.0, # 字权重
                        beta:float = 1.0, # 句子长度系数
                        cutoff_prob:float = 0.99, # 总的概率大于多少截至
                        cutoff_top_n:float = 40, # 总的字符大于多少截至
                        normalization:bool = False, # 是否对音频用预设的方差和均值做标准化，慎用
                        max_batch_size:int = 8, # batch最大数量
                        ) -> None:
        self.max_batch_size = max_batch_size
        # 中英文模型选择
        if lang not in ['cn', 'en']:
            raise Exception("语音识别初始化语言不支持: "+ str(lang))
        self.device = torch.device(device) # 设备
        logging.info("asr_model初始化前总内存占用 %.4f" % (psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024))
        self.asr_model = nemo_asr.models.EncDecCTCModel.restore_from(
                restore_path=model_path, map_location=self.device)
        logging.info("asr_model初始化后总内存占用 %.4f" % (psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024))
        cfg = copy.deepcopy(self.asr_model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False) # Make config overwrite-able
        # 语音特征预处理参数设置
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        self.sample_rate = cfg.preprocessor.sample_rate
        logging.info("模型支持音频采样率为{:d}".format(self.sample_rate))
        if normalization:
            self._cfg_normalization(cfg)
            logging.warning("使用normalization")
        OmegaConf.set_struct(cfg.preprocessor, True) # Disable config overwriting
        # print("preprocessor前总内存占用 %.4f" % (psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024))
        self.asr_model.encoder = self.asr_model.encoder.to(self.device)
        self.asr_model.decoder = self.asr_model.decoder.to(self.device)
        logging.info("asr_model to cuda后总内存占用 %.4f" % (psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024))
        self.preprocessor = nemo_asr.modules.AudioToMelSpectrogramPreprocessor.from_config_dict(cfg.preprocessor)
        logging.info("preprocessor后总内存占用 %.4f" % (psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024))    
        self.data_layer = AudioDataLayer(sample_rate=cfg.preprocessor.sample_rate)
        self.data_loader = DataLoader(self.data_layer, batch_size=max_batch_size, collate_fn=self.data_layer.collate_fn)
        self.wer = WER(
            vocabulary=cfg.decoder.vocabulary,
            batch_dim_index=0,
            use_cer=True,
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=False,
        )
        if use_lm:
            self.beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
                vocab=list(cfg.decoder.vocabulary),
                beam_width=beam_width,
                alpha=alpha, beta=beta,
                cutoff_prob=cutoff_prob,
                cutoff_top_n=cutoff_top_n,
                lm_path=lm_path,
                num_cpus=max(os.cpu_count(), 1),
                input_tensor=False)
        else: self.beam_search_lm = None
        self.asr_model.eval()
        logging.info("总内存占用 %.4f" % (psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024))

    def schedulable_infer(self, input_lists:list):
        return self.infer_wavs(input_lists)[0]

    @torch.no_grad() # 有轻微性能提升，应该是由于数据加载的时候有部分数据没有禁用梯度
    def infer_signal(self, signal:np.ndarray = None):
        """
        对信号进行推理，获取语音识别的结果和概率

        Args:
            signal: 一个一维的numpy数组, 为纯的语音信号
        
        Returns:
            texts: 文本列表
            log_probs: 每一帧音频对应的对数概率值
        """
        self.data_layer.set_signal(signal)
        batch = next(iter(self.data_loader))
        self.data_layer.clear() # 使用后必须清除内存
        audio_signal, audio_signal_len = batch
        # processed_signal: B*E*T, processed_signal_length: B eg:[888, 888]
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_len,
        )
        encoded, encoded_len = self.asr_model.encoder(audio_signal=processed_signal.to(self.device), length=processed_signal_length.to(self.device))
        log_probs = self.asr_model.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        # 一个batch的文本，为一个list: ["文本1", ...]
        texts = self.wer.ctc_decoder_predictions_tensor(
            greedy_predictions, predictions_len=encoded_len, return_hypotheses=False)
        return texts, log_probs

    def infer_wav(self, wav:Union[io.BytesIO, str]=None):
        """
        对音频文件或io对象进行推理，获取语音识别的结果和概率
        Args:
            wav: 一个音频，可以为路径或者一个io对象, 必须为wav格式
        
        Returns:
            texts: 文本列表
            log_probs: 每一帧音频对应的对数概率值
        """
        pre_time = time.time()
        np_audio, secs = self.convert_wave2np(wav)
        logging.info("推理音频时长为{:.3f}s, 处理音频用时为{:.3f}".format(secs, time.time()-pre_time))
        return self.infer_signal(np_audio)

    @staticmethod
    def softmax(logits):
        """
        对对数概率求softmax，用于计算语言模型分数
        """
        result = np.zeros_like(logits)
        for i in range(logits.shape[0]):
            item = logits[i]
            e = np.exp(item - np.max(item))
            result[i] = e / e.sum(axis=-1).reshape([item.shape[0], 1])
        return result
        
    def infer_wavs_withlm(self, wav_ios:Union[List[str], List[io.BytesIO]]=None):
        """
        使用语言模型识别语音文件或者wave io对象。
        """
        if self.beam_search_lm is None:
            raise Exception("语言模型未初始化，请检查是否正确调用语言模型")
        greedy_texts, log_probs,  encoded_len= self.infer_wavs(wav_ios=wav_ios)
        lm_texts = self.beam_search_lm.forward(log_probs = self.softmax(logits=log_probs.cpu().numpy()),
                                                log_probs_length=encoded_len.cpu().numpy())
        return greedy_texts, lm_texts, log_probs
        

    def convert_wave2np(self, wav_io) -> Tuple[np.ndarray, float]:
        """
        将单个音频路径或者io音频对象转换为numpy数组

        Args:
            wav_io: 一个音频，可以为路径或者一个io对象, 必须为wav格式
        
        Returns:
            wav: numpy数组
            secs: 音频的秒数，精确到毫秒
        """
        try:
            pre_time = time.time()
            wav, sr = sf.read(wav_io, always_2d=True)
            logging.debug("音频加载用时:{:.3f}".format(time.time()-pre_time))
            wav = wav[:,0]  # resample
            secs = len(wav) / sr
            if not sr==self.sample_rate: # 如果音频采样率不是目标采样率，就重采样
                target_samples = int(secs * self.sample_rate)  # 重采样后的点数
                wav = scipy.signal.resample(wav, target_samples)  # 重采样的音频,比较耗时
                logging.info("resample用时:{:.3f}".format(time.time()-pre_time))
        except:
            logging.error("音频解析错误！")
            return np.zeros((16000,), dtype=np.float32), 0.1
        # np_audio = (np_audio_float*32768).astype(np.int)
        return wav, secs

    @torch.no_grad()
    def infer_wavs(self, wav_ios:Union[List[str], List[io.BytesIO]]=None) -> Tuple[np.ndarray, List[float]]:
        """
        将多个音频打包为一个batch

        Args:
            wav_ios: 多个音频，可以为路径或者io对象, 音频必须为wav格式
        
        Returns:
            np_audio: numpy数组
            secs: 音频的秒数，精确到毫秒
            encoded_len: 实际长度，用于语言模型解码等
        """
        if wav_ios is None:
            raise Exception("传入批语音数据为空")
        if len(wav_ios) > self.max_batch_size:
            logging.warning("批处理数量超过最大值，将截断，截断数量为{:d}".format(len(wav_ios)-self.max_batch_size))
        pre_time = time.time()
        for wav in wav_ios:
            np_audio, secs = self.convert_wave2np(wav)
            logging.debug("推理音频时长为{:.3f}s".format(secs)) 
            self.data_layer.set_signal(np_audio)
        logging.debug("音频处理总用时{:.3f}".format(time.time()-pre_time))
        batch = next(iter(self.data_loader))
        self.data_layer.clear() # 使用后必须清除内存
        audio_signal, audio_signal_len = batch
        # processed_signal: B*E*T, processed_signal_length: B eg:[888, 888]
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_len,
        )
        encoded, encoded_len = self.asr_model.encoder(audio_signal=processed_signal.to(self.device), length=processed_signal_length.to(self.device))
        log_probs = self.asr_model.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        # 一个batch的文本，为一个list: ["文本1", ...]
        texts = self.wer.ctc_decoder_predictions_tensor(
            greedy_predictions, predictions_len=encoded_len, return_hypotheses=False)
        return texts, log_probs, encoded_len

    def _cfg_normalization(self, cfg):
        """
        设置预设的normalization值
        :param
            cfg: 一个配置项
        """
        normalization = {}
        normalization['fixed_mean'] = [
            -14.95827016, -12.71798736, -11.76067913, -10.83311182,
            -10.6746914,  -10.15163465, -10.05378331, -9.53918999,
            -9.41858904,  -9.23382904,  -9.46470918,  -9.56037,
            -9.57434245,  -9.47498732,  -9.7635205,   -10.08113074,
            -10.05454561, -9.81112681,  -9.68673603,  -9.83652977,
            -9.90046248,  -9.85404766,  -9.92560366,  -9.95440354,
            -10.17162966, -9.90102482,  -9.47471025,  -9.54416855,
            -10.07109475, -9.98249912,  -9.74359465,  -9.55632283,
            -9.23399915,  -9.36487649,  -9.81791084,  -9.56799225,
            -9.70630899,  -9.85148006,  -9.8594418,   -10.01378735,
            -9.98505315,  -9.62016094,  -10.342285,   -10.41070709,
            -10.10687659, -10.14536695, -10.30828702, -10.23542833,
            -10.88546868, -11.31723646, -11.46087382, -11.54877829,
            -11.62400934, -11.92190509, -12.14063815, -11.65130117,
            -11.58308531, -12.22214663, -12.42927197, -12.58039805,
            -13.10098969, -13.14345864, -13.31835645, -14.47345634]
        normalization['fixed_std'] = [
            3.81402054, 4.12647781, 4.05007065, 3.87790987,
            3.74721178, 3.68377423, 3.69344,    3.54001005,
            3.59530412, 3.63752368, 3.62826417, 3.56488469,
            3.53740577, 3.68313898, 3.67138151, 3.55707266,
            3.54919572, 3.55721289, 3.56723346, 3.46029304,
            3.44119672, 3.49030548, 3.39328435, 3.28244406,
            3.28001423, 3.26744937, 3.46692348, 3.35378948,
            2.96330901, 2.97663111, 3.04575148, 2.89717604,
            2.95659301, 2.90181116, 2.7111687,  2.93041291,
            2.86647897, 2.73473181, 2.71495654, 2.75543763,
            2.79174615, 2.96076456, 2.57376336, 2.68789782,
            2.90930817, 2.90412004, 2.76187531, 2.89905006,
            2.65896173, 2.81032176, 2.87769857, 2.84665271,
            2.80863137, 2.80707634, 2.83752184, 3.01914511,
            2.92046439, 2.78461139, 2.90034605, 2.94599508,
            2.99099718, 3.0167554,  3.04649716, 2.94116777]
        cfg.preprocessor.normalize = normalization


if __name__ == '__main__':
    import psutil,os
    logging.basicConfig(level=logging.DEBUG)
    model_path = "checkpoints/QuartzNet15x5Base-Zh.nemo"
    lm_path = "checkpoints/zh_giga.no_cna_cmn.prune01244.klm"
    model = QuartznetModel(model_path=model_path,
                            device="cuda:0", lang="cn", lm_path=lm_path, max_batch_size=10)
    pre_time = time.time()
    print(model.infer_wav("4.wav")[0])
    print("4用时："+ str(time.time() - pre_time))

    pre_time = time.time()
    print(model.infer_wav("2.wav")[0])
    print("2用时："+ str(time.time() - pre_time))

    pre_time = time.time()
    print(model.infer_wav("4.wav")[0])
    print("4用时："+ str(time.time() - pre_time))
    
    pre_time = time.time()
    print(model.infer_wavs(["1.wav", "2.wav", "3.wav", "4.wav", "2.wav", "3.wav", "4.wav","4.wav","4.wav","3.wav"])[0])
    print("pack用时："+ str(time.time() - pre_time))

    pre_time = time.time()
    print(model.infer_wav("1.wav")[0])
    print("4用时："+ str(time.time() - pre_time))

    pre_time = time.time()
    print(str(model.infer_wavs_withlm(["1.wav"])[:-1]))
    print("4用时："+ str(time.time() - pre_time))

    print("内存占用 %.4f" % (psutil.Process(os.getpid()).memory_info().rss /1024 /1024 /1024))

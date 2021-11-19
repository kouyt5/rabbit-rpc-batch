# 语音识别模块

## 模型介绍

基于Quartznet语音识别模型做的封装，模型更多细节请参考(NeMo)[https://github.com/NVIDIA/NeMo]

## 文件下载
从[这里](https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels/files)下载两个模型文件：
+ QuartzNet15x5Base-En.nemo
+ QuartzNet15x5Base-Zh.nemo

分别为中文和英文模型文件。模型初始化时指定路径即可。如果需要语言模型，需要基于kenlm训练

## 部署

1. 在根目录输入`docker-compose up`启动rabbitmq
2. 在本目录下，输入`docker-compose up`，启动算法
3. 启动rabbit-rpc-clent客户端项目
4. 访问接口：
```shell
curl -X POST -F "audio=@/path/to/your/audio.wav" -F "format=wav" http://localhost:8083/asr
```


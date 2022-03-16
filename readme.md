# 基于消息队列的批处理调度
<p align="left">
    <a href="./LICENSE.txt"><img alt="MIT License" src="https://img.shields.io/github/license/kouyt5/rabbit-rpc-batch?style=plastic" /></a>
    <a href="https://www.repostatus.org/#active"><img alt="状态" src="https://www.repostatus.org/badges/latest/active.svg" /></a>
    <a href="https://lgtm.com/projects/g/kouyt5/rabbit-rpc-batch/context:python"><img alt="代码质量" src="https://img.shields.io/lgtm/grade/python/g/kouyt5/rabbit-rpc-batch.svg?logo=lgtm&logoWidth=18"/></a>
    <a href="https://lgtm.com/projects/g/kouyt5/rabbit-rpc-batch/alerts/"><img alt="Total alerts" src="https://img.shields.io/lgtm/alerts/g/kouyt5/rabbit-rpc-batch.svg?logo=lgtm&logoWidth=18"/></a>
</p>

## 项目简介

这是一基于消息队列的批处理调度框架，主要实现了将消息队列中的堆积消息打包给消费者进行处理，并按照顺序返回给生产者，从而在特定任务上提高了系统的吞吐量。该项目位于消息队列中间件和消费者之间，实现消息的按需调度。生产者无法感知到该层的存在。


## Q&A

**为何创建这个项目？**
    
>>
深度学习模型的传统部署基于一对一的方式，例如一个语音识别请求，其中一个音频直接输入到模型然后输出结果。如果用于演示环境，问题不大，但是如果在稍微对性能有一定的需求的场景下，吞吐量将会受限。对于深度学习来说，由于显卡的高效并行方式，往往多个数据一同处理比起单个处理的总体效率更高。因此该项目设计了一个简单框架，用于批处理数据，处理完成后再一个个发送给生产者。对于生产者来说，消息的消费仿佛是一对一一样。经过验证，能够有效的提高吞吐量，在单个http请求的应用场景下，无需更改消息消费逻辑即可无感知的提高系统的处理能力。


**项目总体架构是什么，由哪些部分组成？**

>
项目总体由两个部分组成，消息队列中间件和消费者。生产者目前提供基于SpringBoot的实现 [rabbit-rpc-client](https://github.com/kouyt5/rabbit-rpc-client)，消息队列中间间默认是rabbitmq，关键部分消费者只实现了python端，具体使用请参考 asr 或 se 模块下的示例。

**性能对比**

+ 最大时延 `100ms` (每个请求的最大等待时间)
+ 环境3090
+ 接口包含4个深度学习模型（中英文语音识别和文本分类、实体抽取等）和一些其他业务逻辑
+ batch 每次处理的最大batch数量，无表示没有使用框架


| batch | 并发数 | 平均延迟 |
|----|----|----|
|1|1|276ms|
|12|10|510ms|
|-|10|2332ms|



（注意）`2332ms`为未对语音识别模型优化前测试结果，单条为`463ms`所以真实值可能在1000ms到2000ms之间

## 项目结构
项目基于抽象类设计模块，具有高度可拓展性。
```
.
├── asr  # 语音识别模型实现
│   ├── asr_service.py
│   ├── checkpoints
│   ├── data_layer.py
│   ├── QuartzNetModel.py
├── docker-compose.yml  # rabbitmq
├── locustfile.py  # 性能测试
├── main.py  # 启动文件
├── rpcb  # 调度底层
│   ├── dispatcher  # 批调度器实现
│   ├── message_dispatch.py  # 批调度器基类
│   ├── model.py  # 模型基类
│   ├── rabbit_rpc_consumer.py  # 消息获取模块
│   └── service.py  # 服务基类
├── simple  # 简单的rpc实现
├── tools  # 工具类
```

## 项目启动

+ 启动消息队列： `docker-compose up`
+ 到每个目录下启动算法: `python main.py`
+ 启动client: ...

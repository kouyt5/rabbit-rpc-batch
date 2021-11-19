FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
LABEL author="kouyt5"

# 加速源
RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN  apt-get clean \
    &&apt update -y && apt install libsndfile1 -y
COPY requirements.txt .
RUN pip install -r requirements.txt
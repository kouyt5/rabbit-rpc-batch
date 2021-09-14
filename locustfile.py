from locust import HttpUser, task
import random
import json
import os, time


class ConcurrentTest(HttpUser):

    COUNT = 0
    TOTAL_COUNT = 0
    @task
    def test_asr(self):
        wav_files = [os.path.join('.', 'wav', name) for name in os.listdir('wav')]
        wav_file = random.choice(wav_files)
        wav2trans_dict = {
            '0.wav': "侦察目标搜索海洋一号零翻新",
            '1.wav': "移动目标录入跟踪市长三十三目标高分六号",
            '2.wav': "资源名车搜索目标名称一",
            '3.wav': "资源名称搜索名称路标名称一",
            '4.wav': "区域需求录入名称目标二所伤缴金纬毒东京二十九点一度北为八十一点九度右下角金纬毒为为一百二十点五度东京一百一十七度",
            '5.wav': "目标库录入目标印度太原武术国际机场",
            '6.wav': "下达七百一十三编号的应急快想计划",
            '7.wav': "移动目标需求录入跟踪时长期目标北斗二号导航卫星M二十二星",
            '8.wav': "点路入民生目标名称一经纬度东京一百九点四度为为一百一十九点九度",
            '9.wav': "分起细大妙战的资源",
            '10.wav': "查询安徽省二零二二年十二月十五日下午十一点二十七分的分别率优于六十一点三米的数据",
        }
        trans = wav2trans_dict[wav_file.split('/')[-1]]
        file = {
            "audio": open(wav_file, 'rb')
        }
        data = {"format": 'wav'}
        pre_time = time.time()
        self.TOTAL_COUNT += 1
        with self.client.post("/asr", data=data, files=file, catch_response=True) as response:
            response_data = json.loads(str(response.content, encoding='utf-8'))
            if time.time()-pre_time > 1.:
                self.COUNT += 1
                print("请求超过1000ms, {:d}/{:d}".format(self.COUNT, self.TOTAL_COUNT))
            if response_data["sentence"] == trans:
                response.success()
            else:
                print("输入输出不匹配")
                response.failure('500')
        
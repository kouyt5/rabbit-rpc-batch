from rpcb.service import Service
import threading
import time
import random
import pika
import queue
import logging
import pika.adapters.blocking_connection


"""
消息调度器实现，用于批处理数据，多个数据来了会打包为一个batch输入到模型中，从而提高整体的吞吐量
"""
class BatchMessageDispatcher:
    def __init__(self, callback, max_queue_size:int=32, max_waiting_time:float=0.1,
                    service:Service=None, max_batch_size=10):
        self.safe_queue = queue.Queue(maxsize=max_queue_size)  # 线程安全队列，用于存放数据
        self.message_send_and_ack = callback  # 回调函数
        self.service = service
        guard = GuardThread(max_waiting_time=max_waiting_time,safe_queue=self.safe_queue,
                            compute=self.compute, max_batch_size=max_batch_size)
        guard.start()


    def compute(self, status):
        """
        开始调用算法进行计算
        """
        status = True
        batch = list()
        batch_size = min(10, self.safe_queue.qsize())
        for i in range(batch_size):
            batch.append(self.safe_queue.get())
        # 提取body信息
        service_needed_batch = [item['body'] for item in batch]
        pre_time = time.time()
        result = self.service(service_needed_batch)  # [body1,body2....]
        logging.info("service 用时"+str(time.time()-pre_time))
        # 发送消息到队列中
        for i in range(len(batch)):
            correlation_id = batch[i]['correlation_id']
            reply_to = batch[i]['reply_to']
            delivery_tag = batch[i]['delivery_tag']
            self.message_send_and_ack(result[i], correlation_id, reply_to, delivery_tag)
        status = False
        logging.info("batch size大小："+str(len(result)))
        return result


    def deliver_message(self, body:bytes, properties:pika.BasicProperties, delivery_tag:str) -> None:
        """
        将消息递交给消息管理器管理 
        :params
            body: 传来的数据
            properties: 传来消息的属性，用于回调消息
            delivery_tag: 用于ack答复

        :returns
            None
        """
        message = {'body': body, 'correlation_id':properties.correlation_id,
                    'reply_to':properties.reply_to, 'delivery_tag':delivery_tag}
        self.safe_append_queue(message=message)
        logging.info("添加一个消息入队")
        
    def safe_append_queue(self, message):
        """
        安全的入队
        """
        if self.safe_queue is None:
            raise Exception("消息队列为空")
        if self.safe_queue.full():
            logging.warning(self.safe_queue.qsize())
            # TODO wait until queue is not full
            logging.error("队列长度超出范围")
        self.safe_queue.put(message)


    def check_channel(self):
        """
        检查channel是否可用
        """
        if self.channel is None:
            return False
        return True


class GuardThread(threading.Thread):
    """
    守护进程，用于监听消息是否已经超过指定时间，从而分发数据到服务中
    """
    def __init__(self, max_waiting_time:float=0.1, safe_queue=None, compute=None,max_batch_size=10):
        threading.Thread.__init__(self, daemon=True)
        self.safe_queue = safe_queue
        self.activate = False  # 处理器激活，当有消息来的时候通知消息调度器可以开始处理了，开启一个计时器
        self.is_running = False  # 算法正在处理
        self.max_waiting_time = max_waiting_time  # 最大等待时间，单位秒
        self.max_batch_size = max_batch_size  # 算法最大batch size
        self.start_time = time.time()  # 第一个算法开始时间
        self.compute = compute

    def run(self):
        """
        开始处理消息
        """
        # 监听循环
        while(True):
            # 判断queue是否有数据，如果没有，就继续loop 开始计时，保证一定时间内一定处理
            if self.safe_queue.qsize() == 0:
                self.activate = False
            if (self.safe_queue.qsize() > 0) and not self.is_running:
                # 如果静默状态突然有消息到达，或者运行状态结束队列又存在消息时，开启计时
                if(not self.activate):
                    self.start_time = time.time()
                self.activate = True
                
            # 如果到达时间大于最大等待时间
            if self.activate and time.time()-self.start_time > self.max_waiting_time:
                self.activate = False
                logging.info(" 等待时间："+ str(time.time()-self.start_time))
                # TODO self.is_running 无用，因为计算是阻塞式，计算完成之前，不可能进行新计算，考虑到将来多线程拓展，有必要引入此变量
                self.compute(self.is_running)

            # 如果队列元素超过最大batch size，并且处于非运行状态直接计算
            if self.safe_queue.qsize() >= self.max_batch_size and not self.is_running:
                self.activate = False
                logging.info(" 等待时间："+ str(time.time()-self.start_time))
                self.compute(self.is_running)
            # logging.debug("队列大小"+ str(self.safe_queue.qsize()))
            time.sleep(0.01) # 每隔10ms检测一次

if __name__ == '__main__':
    from base.service import AsrService
    logging.basicConfig(level=logging.DEBUG, format='%(levelname) -10s %(asctime)s %(name) -20s %(funcName) -25s %(lineno) -5d: %(message)s')
    message_disaptcher = BatchMessageDispatcher(None, max_queue_size=32, max_waiting_time=0.1, service=AsrService())
    class Prop:
        correlation_id = "correlation_id"
        reply_to = "reply_to"

    pre_time = time.time()
    for i in range(200):
        message_disaptcher.deliver_message(bytes("hello"+str(i), encoding='utf-8'), properties=Prop(), delivery_tag="delivery_tag")
        sleep_time = float(random.randint(2, 25))/1000
        time.sleep(sleep_time)
    logging.info("计算总时长为: "+ str(time.time()-pre_time))
    
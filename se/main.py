import logging
import sys, os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
from pika.exchange_type import ExchangeType

from torch._C import device
from rpcb.rabbit_rpc_consumer import ReconnectingManager, SelectRabbitConsumer
from rpcb.dispatcher.batch_message_dispatch import BatchMessageDispatcher

from se.SEModel import SEModel
from se.SEService import SEService



def main():
    LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-20s %(lineno) -5d: %(message)s')
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    model_path = "/data/chenc/asr/rabbit-rpc-batch/se/checkpoints/dns48/pretrain.th"
    model = SEModel(model_path=model_path, device="cpu")

    service = SEService(model=model)

    rabbit_consumer = SelectRabbitConsumer(prefech_count=10, exchange='seExchange', exchange_type=ExchangeType.topic, queue='seBase', routing_key='rpc',
                                            host='localhost')
    message_dispatch = BatchMessageDispatcher(rabbit_consumer.message_send_and_ack, max_batch_size=32,
                                        max_waiting_time=0.1, max_queue_size=32,
                                        service=service)
    rabbit_consumer.set_message_dispatcher(message_dispatcher=message_dispatch)
    reconnect_manager = ReconnectingManager(rabbit_consumer)
    reconnect_manager.run()

if __name__ == '__main__':
    main()
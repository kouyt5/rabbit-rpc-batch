from rpcb.message_dispatch import SimpleMessageDispatch, SimpleService
from rpcb.model import ImplDispatchModel
from rpcb.dispatcher.batch_message_dispatch import BatchMessageDispatcher
from rpcb.rabbit_rpc_consumer import ReconnectingManager, SelectRabbitConsumer
from asr.asr_service import AsrService
from asr.QuartzNetModel import QuartznetModel
import logging
from pika.exchange_type import ExchangeType


def main():
    """
    cn
    """
    LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-20s %(lineno) -5d: %(message)s')
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    model_path = "asr/checkpoints/QuartzNet15x5Base-Zh.nemo"
    lm_path = "asr/checkpoints/zh_giga.no_cna_cmn.prune01244.klm"
    model = QuartznetModel(model_path=model_path,
                        device="cpu", lang="cn", lm_path=lm_path, max_batch_size=10, use_lm=False)
    # model = ImplDispatchModel()
    # service = SimpleService()
    service = AsrService(model=model)

    rabbit_consumer = SelectRabbitConsumer(prefech_count=10, exchange='asrExchange', exchange_type=ExchangeType.topic, queue='cc1', routing_key='rpc',
                                            host='localhost')
    message_dispatch = BatchMessageDispatcher(rabbit_consumer.message_send_and_ack, max_batch_size=32,
                                        max_waiting_time=0.1, max_queue_size=32,
                                        service=service)
    rabbit_consumer.set_message_dispatcher(message_dispatcher=message_dispatch)
    reconnect_manager = ReconnectingManager(rabbit_consumer)
    reconnect_manager.run()

def main2():
    LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-20s %(lineno) -5d: %(message)s')
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    model = ImplDispatchModel()
    service = SimpleService()
    # service = AsrService(model=model)

    rabbit_consumer = SelectRabbitConsumer(prefech_count=10, exchange='test', exchange_type=ExchangeType.direct, queue='test', routing_key='test')
    message_dispatch = SimpleMessageDispatch(rabbit_consumer.message_send_and_ack, service=service)
    rabbit_consumer.set_message_dispatcher(message_dispatcher=message_dispatch)
    reconnect_manager = ReconnectingManager(rabbit_consumer)
    reconnect_manager.run()


def main3():
    """
    en
    """
    LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-20s %(lineno) -5d: %(message)s')
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    model_path = "asr/checkpoints/en/QuartzNet15x5Base-En.nemo"
    lm_path = "asr/checkpoints/zh_giga.no_cna_cmn.prune01244.klm"
    model = QuartznetModel(model_path=model_path,
                        device="cuda:0", lang="en", lm_path=lm_path, max_batch_size=10, use_lm=False)
    service = AsrService(model=model)
    rabbit_consumer = SelectRabbitConsumer(prefech_count=10, exchange='asrExchange', exchange_type=ExchangeType.topic, 
                        queue='asr-en', routing_key='asr-en', host='localhost')
    message_dispatch = BatchMessageDispatcher(rabbit_consumer.message_send_and_ack, max_batch_size=32,
                                        max_waiting_time=0.1, max_queue_size=32,
                                        service=service)
    rabbit_consumer.set_message_dispatcher(message_dispatcher=message_dispatch)
    reconnect_manager = ReconnectingManager(rabbit_consumer)
    reconnect_manager.run()

if __name__=="__main__":
    main()
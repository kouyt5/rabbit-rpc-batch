import pika
import json
import msgpack


connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='chenc.icu', port='5672', 
            credentials=pika.PlainCredentials(username='guest', password='guest')))
# connection = pika.SelectConnection(
#     parameters=pika.ConnectionParameters(host='chenc.icu', port='5672', 
#             credentials=pika.PlainCredentials(username='guest', password='guest')),
# )
channel = connection.channel()

# channel.basic_publish()
# msgpack.packb()
# channel.queue_declare(queue='rpc_queue')

def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)

def on_request(ch, method, props, body):
    test = json.loads(body)
    response = {"a": test["a"],"b": test['b']}

    # print(" [.] fib(%s)" % n)
    # response = fib(n)
    exchange = method.exchange
    print(exchange)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id = \
                                                         props.correlation_id),
                     body=bytes(json.dumps(response), encoding='utf-8'))
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_qos(prefetch_count=4)
channel.basic_consume(queue='cc1', on_message_callback=on_request)

print(" [x] Awaiting RPC requests")
channel.start_consuming()
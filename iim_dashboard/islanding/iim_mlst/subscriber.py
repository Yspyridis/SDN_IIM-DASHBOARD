from confluent_kafka import Consumer, KafkaException
import sys
import getopt
import json
import logging
from pprint import pformat


def print_assignment(consumer, partitions):
        print('Assignment:', partitions)


if __name__ == '__main__':

    enum = ['earliest', 'latest']

    # broker = '192.168.21.250:9092'
    broker = '10.100.221.121:9092'
    group = '1'
    offset = 'earliest'
    topics = ['sdnusense_iim']

    conf = {'bootstrap.servers': broker, 
            'group.id': group, 
            'session.timeout.ms': 6000,
            'auto.offset.reset': offset}
            

    logger = logging.getLogger('consumer')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s'))
    logger.addHandler(handler)

    c = Consumer(conf, logger=logger)
    c.subscribe(topics, on_assign=print_assignment)

    # Read messages from Kafka, print to stdout
    try:
        while True:
            msg = c.poll(timeout=1.0)
            if msg is None:
                continue
            else:
                #print(msg.topic())
                #print(msg.partition())
                #print(msg.offset())
                #print(str(msg.key()))
                sys.stderr.write('%s [%d] at offset %d with key %s:\n' %
                                 (msg.topic(), msg.partition(), msg.offset(),
                                  str(msg.key())))
                print(msg.value())

    except KeyboardInterrupt:
        sys.stderr.write('%% Aborted by user\n')

    finally:
        c.close()

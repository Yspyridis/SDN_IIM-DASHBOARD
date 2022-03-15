from confluent_kafka import Consumer, KafkaException
import sys
import getopt
import json
import logging
from pprint import pformat


def print_usage_and_exit(program_name):
    sys.stderr.write('Usage: %s <bootstrap-brokers> <group> <offset> <topic1> <topic2> ..\n' % program_name)
    
    sys.exit(1)


if __name__ == '__main__':

    argv = sys.argv[1:]
    if len(argv) < 3:
        print_usage_and_exit(sys.argv[0])

    enum = ['earliest', 'latest']

    broker = argv[0]
    group = argv[1]
    offset = argv[2]
    if offset not in enum:
        print_usage_and_exit(sys.argv[0])
    topics = argv[3:]

    conf = {'bootstrap.servers': broker, 'group.id': group, 'session.timeout.ms': 6000,
            'auto.offset.reset': offset}

    logger = logging.getLogger('consumer')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s'))
    logger.addHandler(handler)

    c = Consumer(conf, logger=logger)

    def print_assignment(consumer, partitions):
        print('Assignment:', partitions)

    c.subscribe(topics, on_assign=print_assignment)

    # Read messages from Kafka, print to stdout
    try:
        while True:
            msg = c.poll(timeout=1.0)
            if msg is None:
                continue
            else:
                sys.stderr.write('%% %s [%d] at offset %d with key %s:\n' %
                                 (msg.topic(), msg.partition(), msg.offset(),
                                  str(msg.key())))
                print(msg.value())

    except KeyboardInterrupt:
        sys.stderr.write('%% Aborted by user\n')

    finally:
        c.close()

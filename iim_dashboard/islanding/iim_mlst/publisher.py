from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime
import uuid
import json


if __name__ == "__main__":

    # producer = KafkaProducer(bootstrap_servers='[{0}:{1}]'.format('192.168.21.250', 9092))
    producer = KafkaProducer(bootstrap_servers='[{0}:{1}]'.format('10.100.221.121', 9092))

    # Dictionary creation
    event = {
        "name": "ElectricAssetnameOnGridModel",
        "type": "h",
        "category": "DummyValue",
        "gdpr": False,
        "assetId": "23455",
        "riskassessmentId": 1714,
        "businesspartnerId": 1,
        "risklevel": "H",
        "threats": [],
        "cumulativeRiskLevel": "M",
        "contributingirls": [],
        "controls": [],
        "links": [],
        "businessValue": "VH",
        "tags": {}
    }

    # this is necesary.
    timestamp = int(datetime.timestamp(datetime.now()))

    # convert to JSON
    s_json = json.dumps(event)
    print(s_json)

    #  create a producer via helper functions in kafkaHelper file
    try:
        producer.send('sdnusense_iim', s_json.encode('utf-8'), timestamp_ms=timestamp)
        print("Done to try")
    except Exception as e:
        print(e)
        exit(-1)

    try:
        producer.flush()
    except Exception as e:
        print(e)
        exit(-1)

    print("DONE")

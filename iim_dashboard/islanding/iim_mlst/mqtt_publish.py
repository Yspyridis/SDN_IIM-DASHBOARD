from datetime import datetime
import time
import json
import paho.mqtt.client as mqtt_client
import random

MQTT_BROKER = '192.168.1.50'

def print_with_timestamp(string):
    print(datetime.now().__str__() + ": " + string)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.connected_flag = True
        client.bad_connection_flag = False
        print_with_timestamp("Connected sucessfully to MQTT Broker!")
    else:
        client.bad_connection_flag = True
        print_with_timestamp("Failed to connect to MQTT broker, return code %d\n", rc)
        

def on_disconnect(client, userdata, rc):
    print_with_timestamp("disconnecting reason  "  +str(rc))
    client.connected_flag=False
    client.disconnect_flag=True

def main():

    CHANNEL_TOPIC = 'alkyonis-iim-gridmodel'

    # Initialise connection with MQTT
    mqtt_client.Client.connected_flag=False  #create flag in class
    mqtt_client.Client.bad_connection_flag=False  #create flag in class
    mqttClient = mqtt_client.Client('alkyonis-mqtt-iim')
    mqttClient.on_connect = on_connect
    mqttClient.loop_start()
    print_with_timestamp("Connecting to the MQTT broker...")
    mqttClient.connect(MQTT_BROKER, port=1883)
    while not mqttClient.connected_flag: #wait in loop
        print_with_timestamp("In wait loop...")
        time.sleep(1)
    print_with_timestamp("Connection to MQTT sucessfull.")
    
    json_object = {"test": "uc5"}
    json_message = json.dumps(json_object)

    result = mqttClient.publish(CHANNEL_TOPIC, json_message)
    if result[0] == 0:
        print_with_timestamp(f"Record #test sent to MQTT topic `{CHANNEL_TOPIC}`")
    else:
        print_with_timestamp(f"Failed to send message to MQTT topic {CHANNEL_TOPIC}. Error code: {result[0]}")

if __name__ == "__main__":
    main()
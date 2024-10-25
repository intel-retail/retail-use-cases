#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
import cv2
import paho.mqtt.publish as publish
from argparse import ArgumentParser, SUPPRESS

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
from utils.augmentations import letterbox


def process_video(video_path, mqtt_broker_hostname, mqtt_broker_port, mqtt_outgoing_topic):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        results.render()
        print(results)
        detected_products_jsonMsg = results
        #resize frame size
        frame = letterbox(frame, [800,800], stride=32, auto=1)[0]
        cv2.imshow('YOLOv5 Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        
        publish_mqtt_msg(
            detected_products_jsonMsg, mqtt_broker_hostname, mqtt_broker_port,
            mqtt_outgoing_topic
        )

    cap.release()
    cv2.destroyAllWindows()

def publish_mqtt_msg(
        detected_products, mqtt_broker_hostname, mqtt_broker_port, mqtt_outgoing_topic):
    publish.single(
        mqtt_outgoing_topic, str(detected_products),  hostname=mqtt_broker_hostname, port=mqtt_broker_port
    )

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--mqtt', required=True,
                      help='Required. Set mqtt broker host to publish results. Example: 127.0.0.1', type=str)
    args.add_argument('-p', '--port', required=True,
                      help='Required. Set mqtt port to publish results. Example: 1883', type=int)
    args.add_argument('-t', '--topic', required=True,
                      help='Required. Set mqtt topic to publish results. Example: pytorch_yolov5_results', type=str)
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input video present @ performance-tools/sample-media', type=str)
    return parser

if __name__ == '__main__':
    args = build_argparser().parse_args()
    video_path = args.input
    mqtt_broker_hostname = args.mqtt
    mqtt_broker_port = args.port
    mqtt_outgoing_topic = args.topic

    process_video(video_path, mqtt_broker_hostname, mqtt_broker_port, mqtt_outgoing_topic)
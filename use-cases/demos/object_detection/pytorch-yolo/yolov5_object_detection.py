#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

import torch
import cv2
import paho.mqtt.publish as publish

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

if __name__ == '__main__':
    #video_path = '/home/nesubuntu207/Desktop/couple-paying-at-the-counter-in-the-grocery-4121754-3840-15-bench.mp4'
    #video_path = '/home/nesubuntu207/Desktop/coca-cola-4465029-3840-15-bench.mp4'
    video_path = '/home/pipeline-server/sample-media/coca-cola-4465029-1920-15-bench.mp4'
    mqtt_broker_hostname = 'localhost'
    mqtt_broker_port = 1883
    mqtt_outgoing_topic = 'pytorch_yolov5_results'
    process_video(video_path, mqtt_broker_hostname, mqtt_broker_port, mqtt_outgoing_topic)
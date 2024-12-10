#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import cv2
import paho.mqtt.publish as publish
from argparse import ArgumentParser, SUPPRESS
import time

tf_model = tf.keras.applications.ResNet50(weights='imagenet') # nosec
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
model = converter.convert()

# Initialize tflite interpreter
interpreter = tf.lite.Interpreter(model_content=model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def perform_detection(img):

    img = preprocess_input(img)
    
    input_data = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = decode_predictions(output_data, top=2)[0]
    return results

def process_video(video_path, mqtt_broker_hostname, mqtt_broker_port, mqtt_outgoing_topic):
    cap = cv2.VideoCapture(video_path)

    t0 = time.time()
    frame_count = 0
    cumulative_fps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        results = perform_detection(img_resized)
               
        elapsed_time = time.time() - t0
        frame_count += 1
        cumulative_fps = frame_count / elapsed_time
        print(f"Cumulative Average FPS: {cumulative_fps: .2f}")
        print(results)
        for i, (_, label, prob) in enumerate(results):
                if (prob > 0.4):
                    cv2.putText(frame, f"{label}: {prob:.2f}", (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    detected_products_jsonMsg = f"{label}: {prob:.2f}"
                    publish_mqtt_msg(
                        detected_products_jsonMsg, mqtt_broker_hostname, mqtt_broker_port,
                        mqtt_outgoing_topic
                        )

        cv2.imshow('TensorflowLite-Keras Classification', frame)
        if cv2.waitKey(1) == ord('q'):
            break
       
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
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

#!/usr/bin/env python
import io
import os
import re
from pathlib import Path
import tensorflow as tf
import PySimpleGUI as sg
from PIL import Image
import operator
import numpy as np
import cv2
from config import IMG_AUGMENT_LAYERS
from models.slda import SLDA

# Config/Options
from config import IMG_AUGMENT_LAYERS

# Model/Loss definitions
from models.utils import extract_features

# Dataset handling (synthesize/build/query)
from lib.dataset.utils import as_tuple, get_label_distribution
from lib.dataset.synthesizer import synthesize_by_sharding_over_labels
"""
Continual Learning Demo based on PySimpleGUI
--------------------------------------------
This demo is built upon the SLDA algorithms

Known limitations:
1. The GUI only fits well to the laptop screen but unable to scale to a larger monitor
"""

# ------------------------------------------------------------------------------

# to show the learned sku
learned_sku_tracking = {}
learned_sku_tracking_display = []

# to show the testing accuracy
denumerator = 0
top1_numerator = 0
top5_numerator = 0

# ---------SLDA TRAINING -----------------------------------------------------------------------

IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.experimental.AUTOTUNE
"""Choose Model backbone to extract features"""
backbone = tf.keras.applications.EfficientNetV2B0(include_top=False,
                                                  weights='imagenet',
                                                  input_shape=(*IMG_SIZE, 3),
                                                  pooling='avg')
backbone.trainable = False
"""Add augmentation/input layers"""
feature_extractor = tf.keras.Sequential([
    tf.keras.layers.InputLayer(backbone.input_shape[1:]),
    IMG_AUGMENT_LAYERS,
    backbone,
], name='feature_extractor')

feature_extractor.summary()

# SLDA takes a feature vector, linearly maps it to the output class
model = SLDA(n_components=feature_extractor.output_shape[-1], num_classes=100)

# Compile. No loss/optimizer since it is a gradient-free algorithm
model.compile(metrics=['accuracy'])


def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, IMG_SIZE)


def process_path(file_path):
    gt = tf.constant(ground_truth)
    one_hot = gt == list(learned_sku_tracking.keys())
    label = tf.argmax(one_hot)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def Train():
    num_train_samples = 1

    if ground_truth not in learned_sku_tracking:
        learned_sku_tracking[ground_truth] = 0

    learned_sku_tracking[ground_truth] = learned_sku_tracking[ground_truth] + \
        num_train_samples

    # the learned sku for display
    learned_sku_tracking_display.clear()

    for key in learned_sku_tracking:
        temp = [key, learned_sku_tracking[key]]
        learned_sku_tracking_display.append(temp)

    # train per image
    list_ds = tf.data.Dataset.list_files(str(capture_path))

    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    train_features = extract_features(dataset=(labeled_ds.batch(1).prefetch(
        tf.data.AUTOTUNE)), model=feature_extractor)

    N_PARTITIONS = 1

    # This returns a dictionary of partitioned datasets, keyed by partition_id, an integer
    partitioned_dataset = synthesize_by_sharding_over_labels(
        train_features, num_partitions=N_PARTITIONS, shuffle_labels=True)
    # Check the label counts of each partition
    print('Partitions:', len(partitioned_dataset))
    for partition_id in partitioned_dataset:
        dist = get_label_distribution(partitioned_dataset[partition_id])
        print(f'Partition {partition_id}: {dist}')

    # Incrementally train on each partition
    for partition_id in partitioned_dataset:

        print(f'Training [{partition_id+1}/{len(partitioned_dataset)}]')

        # Build Train Dataset pipeline
        train_ds = (
            partitioned_dataset[partition_id].cache().map(
                as_tuple(x='image', y='label'))
            # SLDA learns 1-sample at a time. Inference can be done on batch.
            .batch(1).prefetch(tf.data.AUTOTUNE))

        # SLDA performs well even on a single pass over the dataset
        # model.fit(train_ds, epochs=1, validation_data=test_ds)
        model.fit(train_ds, epochs=1)


def Test(img):

    global denumerator, top1_numerator, top5_numerator

    # image = tf.keras.preprocessing.image.load_img(filename,
    #                                               target_size=(224, 224))

    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(
        img)  # Convert the image to an array

    img_array = tf.expand_dims(img_array, 0)
    m1 = feature_extractor.predict(img_array)
    m2 = model.predict(m1)
    rslt = tf.nn.softmax(m2[0])

    # convert the confidence level into two precision
    two_precision_rslt = list(np.around(np.array(rslt), 2))

    # output the SKU name and the confidence level
    keys_list = learned_sku_tracking.keys()
    values_list = two_precision_rslt
    zip_iterator = zip(keys_list, values_list)
    dictionary = dict(zip_iterator)

    # sort the prediction results according to the softmax probability
    pred_dict = sorted(dictionary.items(),
                       key=operator.itemgetter(1), reverse=True)
    # top five prediction
    pred_dict_top5 = pred_dict[:5]

    # subdirname = os.path.basename(os.path.dirname(filename))

    # extract the labels from the filename string
    pattern = r'[^A-Za-z0-9]+'
    top1_gt = re.sub(pattern, '', str(ground_truth))
    top1_pred = re.sub(
        pattern, '',
        str(pred_dict_top5[0][0]
            if len(pred_dict_top5) >= 2 else "unknown top1"))
    top2_pred = re.sub(
        pattern, '',
        str(pred_dict_top5[1][0]
            if len(pred_dict_top5) >= 2 else "unknown top2"))
    top3_pred = re.sub(
        pattern, '',
        str(pred_dict_top5[2][0]
            if len(pred_dict_top5) >= 3 else "unknown top3"))
    top4_pred = re.sub(
        pattern, '',
        str(pred_dict_top5[3][0]
            if len(pred_dict_top5) >= 4 else "unknown top4"))
    top5_pred = re.sub(
        pattern, '',
        str(pred_dict_top5[4][0]
            if len(pred_dict_top5) >= 5 else "unknown top5"))

    print("Ground Truth: " + top1_gt)
    print("Top-1 Prediction: " + top1_pred)

    # total testing images
    denumerator = denumerator + 1

    # confidence threshold for displaying
    threshold = 0.1

    if top1_gt == top1_pred:
        print("top 1 matched")
        top1_numerator = top1_numerator + 1
        top5_numerator = top5_numerator + 1

    elif top1_gt == top2_pred or top1_gt == top3_pred or top1_gt == top4_pred or top1_gt == top5_pred:
        print("top 5 matched")
        top5_numerator = top5_numerator + 1

    else:
        print("not matched")

    # display the top-5 results
    if (len(pred_dict_top5) >= 1):
        window["Top1"].update(
            pred_dict_top5[0] if pred_dict_top5[0][1] > threshold else '')

    if (len(pred_dict_top5) >= 2):
        window["Top2"].update(
            pred_dict_top5[1] if pred_dict_top5[1][1] > threshold else '')

    if (len(pred_dict_top5) >= 3):
        window["Top3"].update(
            pred_dict_top5[2] if pred_dict_top5[2][1] > threshold else '')

    if (len(pred_dict_top5) >= 4):
        window["Top4"].update(
            pred_dict_top5[3] if pred_dict_top5[3][1] > threshold else '')

    if (len(pred_dict_top5) >= 5):
        window["Top5"].update(
            pred_dict_top5[4] if pred_dict_top5[4][1] > threshold else '')

    print("top1 accuracy ({}/{}) = {} ".format(top1_numerator, denumerator,
                                               top1_numerator / denumerator))
    print("top5 accuracy ({}/{}) = {} ".format(top5_numerator, denumerator,
                                               top5_numerator / denumerator))


webcam_elem = sg.Image(expand_x=True, expand_y=True)

# define layout, show and read the form
col = [[webcam_elem],
       [sg.Frame('Top 1', pad=((0, 0), (50, 0)), layout=[[sg.Text('', size=(15, 3), justification='center', background_color='white', text_color='blue',
                                    font=('Arial', 14), relief='sunken', key='Top1')]]),
        sg.Frame('Top 2', pad=((0, 0), (50, 0)), layout=[[sg.Text('', size=(15, 3), justification='center', background_color='white', text_color='blue',
                                    font=('Arial', 14), relief='sunken', key='Top2')]]),
        sg.Frame('Top 3', pad=((0, 0), (50, 0)), layout=[[sg.Text('', size=(15, 3), justification='center', background_color='white', text_color='blue',
                                    font=('Arial', 14), relief='sunken', key='Top3')]]),
        sg.Frame('Top 4', pad=((0, 0), (50, 0)), layout=[[sg.Text('', size=(15, 3), justification='center', background_color='white', text_color='blue',
                                    font=('Arial', 14), relief='sunken', key='Top4')]]),
        sg.Frame('Top 5', pad=((0, 0), (50, 0)), layout=[[sg.Text('', size=(15, 3), justification='center', background_color='white', text_color='blue',
                                    font=('Arial', 14), relief='sunken', key='Top5')]]),
        ]
       ]


col_files = [[sg.Frame('Learned SKUs', [[sg.Listbox(values='', change_submits=True, size=(
    30, 15), font=('Arial', 14), text_color='blue', key='list_learned_sku')]])],
    [sg.Frame('Ground Truth', [[sg.InputText(size=21, font=("Arial", 20), key="groundtruth")]])],
             [sg.Button('Predict', size=(12, 3), button_color='blue', font=('Arial', 12)),
              sg.Button('Train per image', size=(12, 3), button_color='green', font=('Arial', 12))],
             [sg.Frame('Testing Accuracy',
                       [[sg.Text('', size=(30, 2), justification='center', background_color='white', text_color='blue',
                                 font=('Arial', 14), relief='sunken', key='test-top1accuracy')],
                        [sg.Text('', size=(30, 2), justification='center', background_color='white', text_color='blue',
                                 font=('Arial', 14), relief='sunken', key='test-top5accuracy')]]
                       )
              ]
             ]

col_learned = [[sg.Frame('Learned SKUs', [[sg.Listbox(values='', change_submits=True, size=(
    30, 30), font=('Arial', 14), text_color='blue', key='list_learned_sku')]])]]

# layout = [[sg.Column(col_files), sg.Column(col, vertical_alignment="top"), sg.Column(col_learned)]]
layout = [[sg.Column(col_files), sg.Column(col, vertical_alignment="top")]]


window = sg.Window('Continual Learning Demo',
                   layout,
                   return_keyboard_events=True,
                   location=(0, 0),
                   use_default_focus=False,
                   resizable=True,
                   scaling=1)

# loop reading the user input and displaying image, filename
i = 0

cap = cv2.VideoCapture("/dev/video0")
capture_path = Path("image/capture.png")

# Create capture location if not exists
if not os.path.exists(str(capture_path.parent)):
    os.makedirs(str(capture_path.parent))

while True:
    # read the form
    event, values = window.read(timeout=10)

    # Camera stream
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image)
    bio = io.BytesIO()
    img.save(bio, format='PNG')
    imgbytes = bio.getvalue()

    # perform button and keyboard operations
    if event == sg.WIN_CLOSED:
        break
    elif event == 'Predict':
        # sg.popup(image=imgbytes)
        Test(img)
        window["test-top1accuracy"].update("Top 1 ({}/{}) = {:.2f}%".format(
            top1_numerator, denumerator, top1_numerator * 100 / denumerator))
        window["test-top5accuracy"].update("Top 5 ({}/{}) = {:.2f}%".format(
            top5_numerator, denumerator, top5_numerator * 100 / denumerator))
    elif event == 'Train per image':
        img.save(str(capture_path), format="PNG")
        ground_truth = values['groundtruth']
        Train()
        window["list_learned_sku"].update(learned_sku_tracking_display)
        # os.remove(str(capture_path))

    webcam_elem.update(data=imgbytes)


window.close()

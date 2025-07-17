#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import operator
import re
import av
import threading
from typing import Union
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from model.utils import extract_features
from config import IMG_AUGMENT_LAYERS
from model.slda import SLDA
from lib.dataset.synthesizer import synthesize_by_sharding_over_labels
from lib.dataset.utils import as_tuple, get_label_distribution


st.set_page_config(
    page_title="Continual Learning Demo",
    page_icon="🤖",
    layout="wide",
)

# Override metrics font size, if needed
st.markdown(
    """
<style>
[data-testid="stMetricValue"] {
    font-size: 30px;
}
</style>
""",
    unsafe_allow_html=True,
)

# Session States
if 'counter' not in st.session_state:
    st.session_state['counter'] = 0
if 'sku_tracker' not in st.session_state:
    st.session_state['sku_tracker'] = {}
if 'sku_table' not in st.session_state:
    st.session_state['sku_table'] = []
if 'sku_table_display' not in st.session_state:
    st.session_state['sku_table_display'] = []
if 'img_rgb' not in st.session_state:
    st.session_state['img_rgb'] = None
if 'denumerator' not in st.session_state:
    st.session_state['denumerator'] = 0
if 'top1_numerator' not in st.session_state:
    st.session_state['top1_numerator'] = 0
if 'top5_numerator' not in st.session_state:
    st.session_state['top5_numerator'] = 0
if 'mode' not in st.session_state:
    st.session_state['mode'] = None
if 'mode2_groundtruth' not in st.session_state:
    st.session_state['mode2_groundtruth'] = None

IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    # Choose Model backbone to extract features
    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=(*IMG_SIZE, 3),
        pooling='avg'
    )
    backbone.trainable = False

    # Add augmentation/input layers
    feature_extractor = tf.keras.Sequential([
        tf.keras.layers.InputLayer(backbone.input_shape[1:]),
        IMG_AUGMENT_LAYERS,
        backbone,
    ], name='feature_extractor')

    feature_extractor.summary()

    # SLDA takes a feature vector, linearly maps it to the output class
    model = SLDA(n_components=feature_extractor.output_shape[-1],
                num_classes=100)

    # Compile. No loss/optimizer since it is a gradient-free algorithm
    model.compile(metrics=['accuracy'])

    return model, feature_extractor

def select_folder():
    # Initialize Tkinter for directory chooser
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_path

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, IMG_SIZE)

def process_path(file_path):
    # Get labels
    if st.session_state['mode'] == "file" or st.session_state['mode'] == "folder":
        # Convert the path to a list of path components
        gt = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        one_hot = gt[-2] == list(st.session_state['sku_tracker'].keys())
    elif st.session_state['mode'] == "video-capture":
        gt = tf.constant(st.session_state['mode2_groundtruth'])
        # The second to last is the class-directory
        one_hot = gt == list(st.session_state['sku_tracker'].keys())

    # Integer encode the label
    label = tf.argmax(one_hot)

    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img, label

def Train(mode, abs_path, groundtruth = None):
    global model
    global num_files

    num_train_samples = 1

    # Extract groundtruth from directory
    if mode != "video-capture":
        groundtruth = os.path.basename(os.path.dirname(abs_path))

    print("Abs_path passed to Train():", abs_path)
    matched_files = tf.io.gfile.glob(str(abs_path))
    print("Matched files:", matched_files)

    matched_files = tf.io.gfile.glob(str(abs_path))
    if not matched_files:
        st.error(f"No files found in: {abs_path}")
        return

    # list_ds = tf.data.Dataset.list_files(str(abs_path))
    list_ds = tf.data.Dataset.from_tensor_slices(matched_files)

    # If train per folder, get number of train samples
    if mode == "folder":
        num_train_samples = num_files

    # Prepare SKU tracker
    if groundtruth not in st.session_state['sku_tracker']:
        st.session_state['sku_tracker'][groundtruth] = 0
    st.session_state['sku_tracker'][groundtruth] = st.session_state['sku_tracker'][groundtruth] + num_train_samples
    st.session_state['sku_table'].clear()

    for key in st.session_state['sku_tracker']:
        temp = [key, st.session_state['sku_tracker'][key]]
        st.session_state['sku_table'].append(temp)

    # Set session state for current mode
    st.session_state['mode'] = mode

    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_features = extract_features(dataset=(labeled_ds
                                        .batch(1)
                                        .prefetch(tf.data.AUTOTUNE)), model=feature_extractor)

    print("Extracted features:", train_features)

    N_PARTITIONS = 1

    # This returns a dictionary of partitioned datasets, keyed by partition_id, an integer
    partitioned_dataset = synthesize_by_sharding_over_labels(train_features,
                                                         num_partitions=N_PARTITIONS,
                                                         shuffle_labels=True)

    # Check the label counts of each partition
    # print('Partitions:', len(partitioned_dataset))
    for partition_id in partitioned_dataset:
        dist = get_label_distribution(partitioned_dataset[partition_id])
        print(f'Partition {partition_id}: {dist}')

    for partition_id in partitioned_dataset:
        print(f'Training [{partition_id + 1}/{len(partitioned_dataset)}]')

        # Extract current partition
        partition = (
            partitioned_dataset[partition_id]
            .cache()
            .map(as_tuple(x='image', y='label'))
        )

        # Convert to NumPy arrays
        X_list, y_list = [], []
        for img_tensor, label_tensor in partition:
            try:
                img_np = img_tensor.numpy()
                label_np = label_tensor.numpy()
            except Exception as e:
                print(f"❌ Tensor to NumPy conversion failed: {e}")
                continue

            if img_np.size == 0:
                print("⚠️ Skipping empty image")
                continue

            X_list.append(img_np.squeeze())
            y_list.append(label_np)

        if not X_list:
            print(f"⚠️ Partition {partition_id} is empty, skipping.")
            continue

        X = np.stack(X_list)
        y = np.array(y_list)

        print(f"Training SLDA on X: {X.shape}, y: {y.shape}")
        model.fit(X, y)


def Test(filename, mode, measure_cta):
    image=tf.keras.preprocessing.image.load_img(filename,target_size=(224,224))
    img_array=tf.keras.preprocessing.image.img_to_array(image) # Convert the image to an array
    img_array=tf.expand_dims(img_array,0)
    print("Predicting [1/1]")
    m1 = feature_extractor.predict(img_array)
    m2 = model.predict(m1)
    rslt=tf.nn.softmax(m2[0])

    # convert the confidence level into two precision
    two_precision_rslt = list(np.around(np.array(rslt),2))

    # output the SKU name and the confidence level
    keys_list = st.session_state['sku_tracker'].keys()
    values_list = two_precision_rslt
    zip_iterator = zip(keys_list, values_list)
    dictionary = dict(zip_iterator)

    # sort the prediction results according to the softmax probability
    pred_dict = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    # top five prediction
    pred_dict_top5 = pred_dict[:5]
    print(f"ORIGINAL: {pred_dict_top5}")

    # Confidence threshold
    threshold = 0.05

    # Filter prediction results, remove results < threshold
    filtered_pred_dict_top5 = []
    for pred in pred_dict_top5:
        if pred[1] > threshold:
            filtered_pred_dict_top5.append(pred)
    print(f"FILTERED: {filtered_pred_dict_top5}")

    # extract the labels from the filename string
    if mode == "1":
        subdirname = os.path.basename(os.path.dirname(filename))
        pattern = r'[^A-Za-z0-9]+'
        top1_gt = re.sub(pattern, '', str(subdirname))
        top1_pred = re.sub(pattern, '', str(filtered_pred_dict_top5[0][0] if len(filtered_pred_dict_top5) >= 1 else None))
        top2_pred = re.sub(pattern, '', str(filtered_pred_dict_top5[1][0] if len(filtered_pred_dict_top5) >= 2 else None))
        top3_pred = re.sub(pattern, '', str(filtered_pred_dict_top5[2][0] if len(filtered_pred_dict_top5) >= 3 else None))
        top4_pred = re.sub(pattern, '', str(filtered_pred_dict_top5[3][0] if len(filtered_pred_dict_top5) >= 4 else None))
        top5_pred = re.sub(pattern, '', str(filtered_pred_dict_top5[4][0] if len(filtered_pred_dict_top5) >= 5 else None))

    # Only measure CTA when in mode 1 and measure_cta is True
    if mode == "1" and measure_cta == True:
        st.session_state['denumerator'] = st.session_state['denumerator'] + 1

        if top1_gt == top1_pred:
            print("Top 1 matched")
            st.session_state['top1_numerator'] = st.session_state['top1_numerator'] + 1
            st.session_state['top5_numerator'] = st.session_state['top5_numerator'] + 1
        elif top1_gt == top2_pred or top1_gt == top3_pred or top1_gt == top4_pred or top1_gt == top5_pred:
            print("Top 5 matched")
            st.session_state['top5_numerator'] = st.session_state['top5_numerator'] + 1
        else:
            print("Not matched")

    top1 = filtered_pred_dict_top5[0] if len(filtered_pred_dict_top5) >= 1 else None
    top2 = filtered_pred_dict_top5[1] if len(filtered_pred_dict_top5) >= 2 else None
    top3 = filtered_pred_dict_top5[2] if len(filtered_pred_dict_top5) >= 3 else None
    top4 = filtered_pred_dict_top5[3] if len(filtered_pred_dict_top5) >= 4 else None
    top5 = filtered_pred_dict_top5[4] if len(filtered_pred_dict_top5) >= 5 else None

    print("")
    return top1, top2, top3, top4, top5

def prepare_sku_table():
    st.session_state['sku_table_display'].clear()
    for sku in st.session_state['sku_table']:
        st.session_state['sku_table_display'].append({"sku": sku[0], "qty": sku[1]})


st.title("Continual Learning Demo")
model, feature_extractor = load_model()

# Working mode
st.sidebar.header("Mode")
mode = st.sidebar.selectbox(
    'Choose working mode',
    ('Mode 1 - Local Images', 'Mode 2 - Live Streaming'))

# Mode 1 Variables
selected = None
num_files = 0

if "1" in mode:
    # Choose image folder
    st.sidebar.divider()
    st.sidebar.header("Image Folder")
    selected_folder_path = st.session_state.get("folder_path", None)
    folder_select_button = st.sidebar.button("Select Folder")

    img_viewer_col, sku_col = st.columns([0.7, 0.3])
    img_viewer_col.header("Image Viewer")

    if folder_select_button:
        selected_folder_path = select_folder()
        st.session_state.folder_path = selected_folder_path

    if selected_folder_path:
        st.sidebar.caption(f"Selected folder path: {selected_folder_path}")
        filelist=[]
        for root, dirs, files in os.walk(selected_folder_path):
            for file in files:
                    filename=os.path.join(root, file)
                    _, ext = os.path.splitext(file)
                    if ext.lower() in (".png", ".jpg", "jpeg", ".tiff", ".bmp"):
                        filelist.append({"filename": file})
        num_files = len(filelist)

        # Image file browser
        st.sidebar.divider()
        df = pd.DataFrame.from_dict(filelist)
        gb = GridOptionsBuilder.from_dataframe(df)
        selection_mode = "single"
        gb.configure_selection(selection_mode, use_checkbox=False)
        gridOptions = gb.build()
        st.sidebar.header("Image Picker")
        with st.sidebar:
            grid_response = AgGrid(
                df,
                gridOptions=gridOptions,
                width='100%',
                height=250,
                fit_columns_on_grid_load=True,
                allow_unsafe_jscode=True,
                enable_enterprise_modules=False,
            )
        selected = grid_response['selected_rows']
        selected_df = pd.DataFrame(selected).apply(pd.to_numeric, errors='coerce')
    else:
        st.info('Select image folder from sidebar pane.', icon="ℹ️")

    # If selected folder has files, show Predict & Train Pane
    if num_files >= 1:
        # Predict & Train Pane
        st.sidebar.divider()
        st.sidebar.header("Predict & Train",
                          help="In Mode 1, the ground truth is defined as the name of the directory containing the image.")
        predict_train_container = st.sidebar.container()
        predict_placeholder = predict_train_container.empty()
        tas_checkbox_placeholder = predict_train_container.empty()
        groundtruth_placeholder = predict_train_container.empty()
        train_image_placeholder = predict_train_container.empty()
        train_folder_btn = st.sidebar.button("Train per folder", key="train_folder_btn")
        auto_groundtruth = os.path.basename(os.path.dirname(filename))
        train_image_btn = None

        # If train by folder
        if train_folder_btn:
            files = f"{selected_folder_path}/*"
            Train(mode="folder", abs_path=files)

        # If file is clicked
        if selected:
            # Auto-populate groundtruth from folder and set as text input
            # Textbox disabled due to Mode 1
            groundtruth = groundtruth_placeholder.text_input('Ground Truth', auto_groundtruth if auto_groundtruth else None, disabled=True)

            # Show predict and train by image button
            predict_btn = predict_placeholder.button("Predict", key="predict_btn")
            train_image_btn = train_image_placeholder.button("Train per image", key="train_image_btn")

            # Image Viewer
            image = Image.open(f"{selected_folder_path}/{selected[0]['filename']}")
            img_viewer_col.image(image, caption=selected[0]['filename'])

            # If train by image
            if train_image_btn:
                file = f"{selected_folder_path}/{selected[0]['filename']}"
                Train(mode="file", abs_path=file)

            # If run prediction
            if predict_btn:
                file = f"{selected_folder_path}/{selected[0]['filename']}"

                top1, top2, top3, top4, top5 = Test(filename=file, mode="1", measure_cta=True if len(st.session_state['sku_table']) > 1 else False)

                # Show prediction Result
                st.divider()
                st.header("Prediction Results")
                top1_col, top2_col, top3_col, top4_col, top5_col = st.columns(5)
                top1_col.metric(label="Top 1", value=str(top1[0] if top1 else "N/A"))
                top2_col.metric(label="Top 2", value=str(top2[0] if top2 else "N/A"))
                top3_col.metric(label="Top 3", value=str(top3[0] if top3 else "N/A"))
                top4_col.metric(label="Top 4", value=str(top4[0] if top4 else "N/A"))
                top5_col.metric(label="Top 5", value=str(top5[0] if top5 else "N/A"))

                # st.write("Predicted:", top1[0])
                # st.write("Ground Truth:", auto_groundtruth)

                # Cumulative Testing Accuracy
                st.divider()
                st.header("Cumulative Testing Accuracy", help=f"Requires at least 2 classes / SKUs. Top-1 indicates the accuracy where the network has predicted the correct label with highest probability. Top-5 indicates the accuracy where the correct label appears in the network's top five predicted classes.")

                if len(st.session_state['sku_table']) > 1:
                    test1_col, test5_col = st.columns(2)

                    test1_col.metric(label="Top 1", value=f"{st.session_state['top1_numerator']}/{st.session_state['denumerator']} = {st.session_state['top1_numerator'] * 100 / st.session_state['denumerator']:0.2f}%")
                    test5_col.metric(label="Top 5", value=f"{st.session_state['top5_numerator']}/{st.session_state['denumerator']} = {st.session_state['top5_numerator'] * 100 / st.session_state['denumerator']:0.2f}%")
                else:
                    st.warning('Cumulative Testing Accuracy is not available when only 1 class / SKU present.', icon="⚠️")

        # If image is not selected in Image Browser
        else:
            img_viewer_col.info('To view an image, simply select it from the image picker located on the sidebar.', icon="ℹ️")
            groundtruth = groundtruth_placeholder.text_input('Ground Truth', auto_groundtruth, disabled=True)

        # Display SKU table
        if st.session_state['sku_table'] != []:
            sku_col.header("Learned SKU")
            with sku_col:
                prepare_sku_table()
                df_sku_table = pd.DataFrame.from_dict(st.session_state['sku_table_display'])
                gb = GridOptionsBuilder.from_dataframe(df_sku_table)
                selection_mode = "single"
                gb.configure_selection(selection_mode, use_checkbox=False)
                gridOptions = gb.build()
                grid_response = AgGrid(
                    df_sku_table,
                    gridOptions=gridOptions,
                    width='100%',
                    height=400,
                    fit_columns_on_grid_load=True,
                    allow_unsafe_jscode=True,
                    enable_enterprise_modules=False,
                )
                selected_sku = grid_response['selected_rows']

elif "2" in mode:
    class VideoProcessor(VideoProcessorBase):
        # Create lock object for thread-safety
        frame_lock: threading.Lock
        in_image: Union[np.ndarray, None]
        out_image: Union[np.ndarray, None]

        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            self.in_image = None
            self.out_image = None

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            in_image = frame.to_ndarray(format="rgb24")

            # Image flipping. Depends on camera mounting configuration
            out_image = in_image[:, ::-1, :]

            with self.frame_lock:
                self.in_image = in_image
                self.out_image = out_image

            # TODO: Check which frame to pass
            return av.VideoFrame.from_ndarray(in_image)

    img_viewer_col, sku_col = st.columns([0.7, 0.3])

    with img_viewer_col:
        st.header("Camera WebRTC Stream")
        ctx = webrtc_streamer(key="snapshot",
                        video_processor_factory=VideoProcessor,
                        media_stream_constraints={"audio": False, "video": True})

    if ctx.video_processor:
        # Predict & Train Pane
        st.sidebar.divider()
        st.sidebar.header("Predict & Train")

        # Layout
        predict_train_container = st.sidebar.container()
        predict_btn = predict_train_container.button("Predict", key="predict_btn")
        groundtruth = predict_train_container.text_input('Ground Truth')
        train_btn = predict_train_container.button("Train", key="train_btn")

        # Prediction
        if predict_btn:
            with ctx.video_processor.frame_lock:
                in_image = ctx.video_processor.in_image
                out_image = ctx.video_processor.out_image

            if in_image is not None and out_image is not None:
                # Change to out_image if image flipping is required
                im = Image.fromarray(in_image)
                im.save("./images/frame.jpg")
                top1, top2, top3, top4, top5 = Test(filename="./images/frame.jpg", mode="2", measure_cta=False)

                # Remove file after training
                if os.path.exists("./images/frame.jpg"):
                    os.remove("./images/frame.jpg")

                # Show prediction Result
                st.divider()
                st.header("Prediction Results")
                top1_col, top2_col, top3_col, top4_col, top5_col = st.columns(5)
                top1_col.metric(label="Top 1", value=str(top1))
                top2_col.metric(label="Top 2", value=str(top2))
                top3_col.metric(label="Top 3", value=str(top3))
                top4_col.metric(label="Top 4", value=str(top4))
                top5_col.metric(label="Top 5", value=str(top5))

            else:
                st.warning("No frames available yet.")

        # Training
        if train_btn and groundtruth:
            with ctx.video_processor.frame_lock:
                in_image = ctx.video_processor.in_image
                out_image = ctx.video_processor.out_image

            if in_image is not None and out_image is not None:
                # Change to out_image if image flipping is required
                im = Image.fromarray(in_image)
                im.save("./images/frame.jpg")
                st.session_state['mode2_groundtruth'] = groundtruth
                Train(mode="video-capture", abs_path="./images/frame.jpg", groundtruth=groundtruth)

                # Remove file after training
                if os.path.exists("./images/frame.jpg"):
                    os.remove("./images/frame.jpg")
            else:
                st.warning("No frames available yet.")
        # Block if attempt to train without ground truth
        elif train_btn and groundtruth == "":
            st.sidebar.error("Ground Truth not specified!")

    # Display SKU table
    if st.session_state['sku_table'] != []:
        sku_col.header("Learned SKU")
        with sku_col:
            prepare_sku_table()
            df_sku_table = pd.DataFrame.from_dict(st.session_state['sku_table_display'])
            gb = GridOptionsBuilder.from_dataframe(df_sku_table)
            selection_mode = "single"
            gb.configure_selection(selection_mode, use_checkbox=False)
            gridOptions = gb.build()
            grid_response = AgGrid(
                df_sku_table,
                gridOptions=gridOptions,
                width='100%',
                fit_columns_on_grid_load=True,
                allow_unsafe_jscode=True,
                enable_enterprise_modules=False,
            )
            selected_sku = grid_response['selected_rows']

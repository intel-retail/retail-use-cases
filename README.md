# Retail Use Cases

![CodeQL](https://github.com/intel-retail/automated-self-checkout/actions/workflows/codeql.yaml/badge.svg?branch=main)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/intel-retail/automated-self-checkout/badge)](https://api.securityscorecards.dev/projects/github.com/intel-retail/automated-self-checkout)

There are various use cases available, click one each hyperlink to find out more details:
- [dlstreamer gst](#dlstreamer-gst-use-case)
- [grpc python](#grpc_python-use-case)
- [gst c-api](#gst_capi-use-cases)
- [demos grpc OVMS](#demos-use-case)
- [grpc go](#grpc_go-use-case)
- [pytorch object detection use case](#pytorch-object-detection-use-case)
- [tensorflow keras classification use case](#tensorflow-keras-classification-use-case)
---
---

## dlstreamer gst use case:

### build dlstreamer gst Docker image

- to build run:

    ```bash
    make build-gst
    ```

### run dlstreamer gst use cases

- run one gst dlstreamer pipeline and use the default object detection only (yolov5s.sh)

    ```bash
    make run-gst
    ```

- run three gst dlstreamer pipelines and use the default object detection only (yolov5s.sh)

    ```bash
    PIPELINE_COUNT=3 make run-gst
    ```

- run two gst dlstreamer pipelines and use the object detection with classification (yolov5s_effnetb0.sh)

    ```bash
    PIPELINE_SCRIPT=yolov5s_effnetb0.sh PIPELINE_COUNT=2 make run-gst
    ```

- shutdown Docker containers

    ```bash
    make down-gst
    ```

- clean up the output results

    ```bash
    make clean-results
    ```

---

## grpc_python use case:

### build grpc_python Docker image

- to build run:

    ```bash
    make build-grpc_python
    ```

### run grpc_python use cases

- run one grpc_python pipeline and use the default model (instance-segmentation-security-1040)

    ```bash
    make run-grpc_python
    ```
- show the supported MODEL_NAME for grpc_python

    ```bash
    make list-grpc-python-model-names
    ```

- run three grpc_python pipelines and use yolov5s model

    ```bash
    PIPELINE_COUNT=3 MODEL_NAME=yolov5s make run-grpc_python
    ```

- shutdown Docker containers

    ```bash
    make down-grpc_python
    ```

- clean up the output results

    ```bash
    make clean-results
    ```

---

## gst_capi use cases:

There are three different pipelines for gst_capi use cases: capi_yolov5, capi_yolov5_ensemble, and capi_face_detection.

### build all capi pipeline Docker images

- to build all capis run:

    ```bash
    make build-all-capis
    ```

- to build one at a time run:

    - for capi_yolov5

        ```bash
        make build-capi_yolov5
        ```

    - for capi_yolov5_ensemble

        ```bash
        make build-capi_yolov5_ensemble
        ```

    - for capi_face_detection

        ```bash
        make build-capi_face_detection
        ```

### run gst_capi use cases

- run one capi_yolov5 pipeline

    ```bash
    make run-capi_yolov5
    ```

- shutdown capi_yolov5

    ```bash
    make down-capi_yolov5
    ```

- run one capi_yolov5_ensemble pipeline

    ```bash
    make run-capi_yolov5_ensemble
    ```

- shutdown capi_yolov5_ensemble

    ```bash
    make down-capi_yolov5_ensemble
    ```

- run one capi_face_detection pipeline

    ```bash
    make run-capi_face_detection
    ```

- shutdown capi_face_detection

    ```bash
    make down-capi_face_detection
    ```

- run two capi_yolov5_ensemble pipelines

    ```bash
    PIPELINE_COUNT=2 make run-capi_yolov5_ensemble
    ```

- shutdown Docker containers

    ```bash
    make down-capi_yolov5
    make down-capi_yolov5_ensemble
    make down-capi_face_detection
    ```

- clean up the output results

    ```bash
    make clean-results
    ```

---

## demos use case:

There are three different pipelines for demos use case: classification, instance_segmentation, and object_detection.

### build all demos Docker image

- to build run:

    ```bash
    make build-demos
    ```

### run demos use case

- run one classification pipeline

    ```bash
    make run-demo-classification
    ```
- run one instance segmentation pipeline

    ```bash
    make run-demo-instance-segmentation
    ```

- run two object detection pipelines

    ```bash
    PIPELINE_COUNT=2 make run-demo-object-detection
    ```

- shutdown classification pipeline Docker container

    ```bash
    make down-demo-classification
    ```

- shutdown all Docker containers

    ```bash
    make down-demos-all
    ```

- clean up the output results

    ```bash
    make clean-results
    ```

---

## grpc_go use case:

### build grpc_go Docker image

- to build run:

    ```bash
    make build-grpc-go
    ```

### run grpc_go use cases

- run one grpc_go pipeline

    ```bash
    make run-grpc-go
    ```

- run two grpc_go pipelines

    ```bash
    PIPELINE_COUNT=2 make run-grpc-go
    ```

- shutdown Docker containers

    ```bash
    make down-grpc-go
    ```

- clean up the output results

    ```bash
    make clean-results
    ```

## pytorch object detection use case:

There are two versions of this pipeline implemented in the code -

- Using YOLOv5 from [PyTorch Hub](https://pytorch.org/hub/)
- Using YOLOv5 from [Ultralytics](https://docs.ultralytics.com/)

**Intel Optimized packages for pytorch is available only for running inference using OpenVino for Ultralytics implementation.**

Performance comparison between using Intel optimzed packages for Ultralytics using OpenVino on different Intel platforms is available on the [official Ultralytics Page](https://docs.ultralytics.com/integrations/openvino/)

As a developer make sure to set these two parameters in [pytorch-yolov5-env.txt](use-cases/demos/pytorch-yolov5-env.txt) -

- **USE_ULTRALYTICS** - Set it to 1 to use Ultralytics pipeline else to 0
- **INTEL_OPTIMIZED** - Set it to 1 to use Intel optimzed packages for Ultralytics implementation else to 0

USE_ULTRALYTIC = 0 - use torch hub implementation (doesn't matter what INTEL_OPTIMIZED is )

USE_ULTRALYTIC = 1, INTEL_OPTIMIZED = 0 - use ultralytics implementation

USE_ULTRALYTIC = 1, INTEL_OPTIMIZED = 1 - use ultralytics implementation with intel optimized packages

**Note: Default for both these are set as 0.**

Run the following commands by changing directory to ./retail-use-cases/use-cases/demos

- build pytorch pipeline

    ```bash
    make build_pytorch
    ```

- run one pytorch pipeline

    ```bash
    make run_pytorch_object_detection
    ```

- run two pytorch pipelines

    ```bash
    PIPELINE_COUNT=2 make run_pytorch_object_detection
    ```

- shutdown Docker containers

    ```bash
    make down_pytorch_object_detection
    ```

Run the following command by changing directory to ./retail-use-cases

- clean up the output results

    ```bash
    make clean-results

## tensorflow keras classification use case:

Run the following commands by changing directory to ./retail-use-cases/use-cases/demos

- build tensorflow keras pipeline

    ```bash
    make build_tensorflow_keras
    ```

- run one tensorflow keras pipeline

    ```bash
    make run_tensorflow_keras_classification
    ```

- run two tensorflow keras pipelines

    ```bash
    PIPELINE_COUNT=2 make run_tensorflow_keras_classification
    ```

- shutdown Docker containers

    ```bash
    make down_tensorflow_keras_classification
    ```

Run the following command by changing directory to ./retail-use-cases

- clean up the output results

    ```bash
    make clean-results

---

## Disclaimer

GStreamer is an open source framework licensed under LGPL. See https://gstreamer.freedesktop.org/documentation/frequently-asked-questions/licensing.html?gi-language=c.  You are solely responsible for determining if your use of Gstreamer requires any additional licenses.  Intel is not responsible for obtaining any such licenses, nor liable for any licensing fees due, in connection with your use of Gstreamer.

Certain third-party software or hardware identified in this document only may be used upon securing a license directly from the third-party software or hardware owner. The identification of non-Intel software, tools, or services in this document does not constitute a sponsorship, endorsement, or warranty by Intel.

## Datasets & Models Disclaimer:

To the extent that any data, datasets or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality.  By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license.

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets or models.

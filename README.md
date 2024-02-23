# Retail Use Cases

## example run commands from profile-launcher (WILL BE DEPRECATED)

```bash
./core-services/profile-launcher/profile-launcher --configdir ./use-cases/dlstreamer/res --inputsrc /dev/video4 --target_device CPU

./core-services/profile-launcher/profile-launcher --configdir ./use-cases/grpc_python/res --inputsrc /dev/video4 --target_device CPU
```


## build dlstreamer gst Docker image

(cd use-cases/dlstreamer; make build)


## run dlstreamer gst use cases

### example run commands

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



## Disclaimer

GStreamer is an open source framework licensed under LGPL. See https://gstreamer.freedesktop.org/documentation/frequently-asked-questions/licensing.html?gi-language=c.  You are solely responsible for determining if your use of Gstreamer requires any additional licenses.  Intel is not responsible for obtaining any such licenses, nor liable for any licensing fees due, in connection with your use of Gstreamer.

Certain third-party software or hardware identified in this document only may be used upon securing a license directly from the third-party software or hardware owner. The identification of non-Intel software, tools, or services in this document does not constitute a sponsorship, endorsement, or warranty by Intel.

## Datasets & Models Disclaimer:

To the extent that any data, datasets or models are referenced by Intel or accessed using tools or code on this site such data, datasets and models are provided by the third party indicated as the source of such content. Intel does not create the data, datasets, or models, provide a license to any third-party data, datasets, or models referenced, and does not warrant their accuracy or quality.  By accessing such data, dataset(s) or model(s) you agree to the terms associated with that content and that your use complies with the applicable license.

Intel expressly disclaims the accuracy, adequacy, or completeness of any data, datasets or models, and is not liable for any errors, omissions, or defects in such content, or for any reliance thereon. Intel also expressly disclaims any warranty of non-infringement with respect to such data, dataset(s), or model(s). Intel is not liable for any liability or damages relating to your use of such data, datasets or models.

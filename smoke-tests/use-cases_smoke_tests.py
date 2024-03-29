#!/usr/bin/env python3
"""
 Copyright (C) 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import sys
import subprocess  # nosec
import time
import glob


def run_command(command, envs=None):
    try:
        if envs is None:
            subprocess.run(command, shell=True)  # nosec
        else:
            subprocess.run(command, env=envs, shell=True)  # nosec
    except subprocess.CalledProcessError as ex:
        print(f'Command {ex.cmd} failed with error {ex.returncode}')


def clean_results():
    run_command(" ".join(["make", "clean-results"]))


def build_use_case(build_target):
    run_command(" ".join(["make", build_target]))


def run_use_case(run_target, envs=None):
    run_command(" ".join(["make", run_target]), envs)


def down_use_case(down_target, envs=None):
    run_command(" ".join(["make", down_target]), envs)
    time.sleep(10)


def checkForNonEmptyPipelineLogFiles(test_case_name, result_dir, file_pattern):
    max_wait_time = 300
    sleep_increments = 10
    total_wait_time = 0
    while True:
        matching_files = glob.glob(os.path.join(result_dir, file_pattern))
        if all([os.path.isfile(file) and
                os.path.getsize(file) > 0 for file in matching_files]):
            print(
               f'=== test case [{test_case_name}] '
               'smoke test pipeline log PASSED')
            break
        elif total_wait_time > max_wait_time:
            print(
               f'FAILED: exceeding the max wait time {max_wait_time} '
               'while waiting for log files, stop waiting')
            print(
               f'=== test case [{test_case_name}] '
               'smoke test pipeline log FAILED')
            return
        else:
            print(
               f'could not find all matching log files yet, '
               'sleep for {sleep_increments} seconds and retry it again')
            time.sleep(sleep_increments)
            total_wait_time += sleep_increments
    print(f"total wait time = {total_wait_time} seconds")


def test_dlstreamer(envs):
    clean_results()
    build_use_case('build-gst')
    print("starting dlstreamer use case:")
    # PIPELINE_COUNT=2 make run-gst
    envs["PIPELINE_COUNT"] = "2"
    run_use_case('run-gst', envs)
    checkForNonEmptyPipelineLogFiles(
        'dlstreamer with 2 pipelines',
        './results', 'pipeline*_gst.log')
    down_use_case('down-gst', envs)
    clean_results()
    # PIPELINE_SCRIPT=yolov5s_effnetb0.sh PIPELINE_COUNT=2 make run-gst
    envs["PIPELINE_SCRIPT"] = "yolov5s_effnetb0.sh"
    run_use_case('run-gst', envs)
    checkForNonEmptyPipelineLogFiles(
        'dlstreamer with classification '
        'efficientNet 2 pipelines',
        './results', 'pipeline*_gst.log')
    down_use_case('down-gst', envs)


def test_grpc_python(envs):
    clean_results()
    build_use_case('build-grpc_python')
    print("starting grpc_python use case:")
    # PIPELINE_COUNT=2 make run-grpc_python
    envs["PIPELINE_COUNT"] = "2"
    run_use_case('run-grpc_python', envs)
    checkForNonEmptyPipelineLogFiles(
        'grpc_python with default model 2 pipelines',
        './results',
        'pipeline*_grpc_python.log')
    down_use_case('down-grpc_python', envs)
    clean_results()
    # PIPELINE_COUNT=2 MODEL_NAME=yolov5s make run-grpc_python
    envs["MODEL_NAME"] = "yolov5s"
    run_use_case('run-grpc_python', envs)
    checkForNonEmptyPipelineLogFiles(
        'grpc_python with model '
        'yolov5s 2 pipelines', './results',
        'pipeline*_grpc_python.log')
    down_use_case('down-grpc_python', envs)


def test_gst_capi(envs):
    clean_results()
    build_use_case('build-capi_yolov5')
    print("starting gst-capi_yolov5 use case:")
    run_use_case('run-capi_yolov5')
    checkForNonEmptyPipelineLogFiles(
        'gst-capi_yolov5',
        './results',
        'pipeline*_capi_yolov5.log')
    down_use_case('down-capi_yolov5')
    clean_results()
    build_use_case('build-capi_yolov5_ensemble')
    print("starting gst-capi_yolov5_ensemble use case:")
    envs["PIPELINE_COUNT"] = "2"
    run_use_case('run-capi_yolov5_ensemble', envs)
    checkForNonEmptyPipelineLogFiles(
        'gst-capi_yolov5_ensemble with 2 pipelines',
        './results',
        'pipeline*_capi_yolov5_ensemble.log')
    down_use_case('down-capi_yolov5_ensemble')
    clean_results()
    build_use_case('build-capi_face_detection')
    print("starting gst-capi_face_detection use case:")
    run_use_case('run-capi_face_detection')
    checkForNonEmptyPipelineLogFiles(
        'gst-capi_face_detection',
        './results',
        'pipeline*_capi_face_detection.log')
    down_use_case('down-capi_face_detection')


def test_demos(envs):
    clean_results()
    build_use_case('build-demos')
    print('starting demos use case:')
    run_use_case('run-demo-classification')
    checkForNonEmptyPipelineLogFiles(
        'demo-classification',
        './results',
        'pipeline*_demo_classification.log')
    run_use_case('run-demo-instance-segmentation')
    checkForNonEmptyPipelineLogFiles(
        'demo-instance-segmentation',
        './results',
        'pipeline*_demo_instance_segmentation.log')
    down_use_case('down-demo-classification')
    envs["PIPELINE_COUNT"] = "2"
    run_use_case('run-demo-object-detection', envs)
    checkForNonEmptyPipelineLogFiles(
        'demo-object-detection',
        './results',
        'pipeline*_demo_object_detection.log')
    down_use_case('down-demos-all')


def test_grpc_go(envs):
    clean_results()
    build_use_case('build-grpc-go')
    print('starting grpc_go use case:')
    run_use_case('run-grpc-go')
    checkForNonEmptyPipelineLogFiles(
        'grpc_go',
        './results',
        'pipeline*_grpc_go.log')
    down_use_case('down-grpc-go')
    clean_results()
    envs["PIPELINE_COUNT"] = "2"
    run_use_case('run-grpc-go', envs)
    checkForNonEmptyPipelineLogFiles(
        'grpc_go with 2 pipelines',
        './results',
        'pipeline*_grpc_go.log')
    down_use_case('down-grpc-go')


def main():
    print("retail-use-cases smoke testing starting...")
    run_command('pwd')
    env_vars = os.environ.copy()
    print("TEST_DIR:", env_vars["TEST_DIR"])
    test_dir = env_vars["TEST_DIR"]
    os.chdir(test_dir)
    print("current directory is:", os.getcwd())
    # clean all just in case there are some residuals
    clean_results()
    run_command('docker rm $(docker ps -aq) -f')
    # starting test cases:
    test_dlstreamer(env_vars)
    test_grpc_python(env_vars)
    test_gst_capi(env_vars)
    test_demos(env_vars)
    test_grpc_go(env_vars)

    clean_results()


if __name__ == '__main__':
    sys.exit(main() or 0)

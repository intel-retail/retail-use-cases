#!/bin/bash
#
# Copyright (C) 2024 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0
#

GRPC_PORT="${GRPC_PORT:=9001}"
OVMS_INIT_TIME_IN_SECOND="${OVMS_INIT_TIME_IN_SECOND:=10}"

# getFreePortNumber scanning through some port number range and returns a port number that is not currently being used
function getFreePortNumber() {
    startPort=8080
    lastPort=8999
    displayPortNum=
    # find out the current free port number from a given port number range
    for ((checkPort=$startPort; checkPort<=$lastPort; checkPort++))
    do
        portUsed=0
        netstat -vatn | grep -e ":$checkPort[^[:alnum:]]">/dev/null && portUsed=1
        echo >&2 "portUsed= $portUsed"

        if [ "$portUsed" == 0 ]
        then
            echo >&2 "$checkPort is free to use"
            displayPortNum=$checkPort
            break
        fi
    done

    if [ -z "$displayPortNum" ]
    then
        echo >&2 "could not find the free port number for display, exit"
        exit 1
    fi

    echo "$displayPortNum"
}

displayPortNum=$( getFreePortNumber )

# this timing is to wait for ovms server models being ready
sleep $OVMS_INIT_TIME_IN_SECOND

PROFILE_NAME="grpc_go"

echo "running $PROFILE_NAME with displayPortNum=$displayPortNum"

# generate unique container id based on the date with the precision upto nano-seconds
cid=$(date +%Y%m%d%H%M%S%N)
echo "cid: $cid"

if [ -n "$DEBUG" ]
then
    ./grpc-go -i $INPUTSRC -u 127.0.0.1:$GRPC_PORT -h 0.0.0.0:$displayPortNum
else
    ./grpc-go -i $INPUTSRC -u 127.0.0.1:$GRPC_PORT -h 0.0.0.0:$displayPortNum 2>&1  | tee >/tmp/results/r"$cid"_"$PROFILE_NAME".jsonl >(stdbuf -oL sed -n -e 's/^.*fps: //p' | stdbuf -oL cut -d , -f 1 > /tmp/results/pipeline"$cid"_"$PROFILE_NAME".log)
fi

hasError5=$(grep "ERROR(5)-" /tmp/results/r"$cid"_"$PROFILE_NAME".jsonl)
MAX_RETRIES=10
retry_cnt=1
# Due to the race conditioin from replica multiple instances of docker-compose; the port binding could still fail
# will rerty it again
while [ -n "$hasError5" ]
do
    echo "port binding failed, retry_cnt=$retry_cnt"
    if [ "$retry_cnt" -gt "$MAX_RETRIES" ]
    then
        echo "already reach maximum retries: $MAX_RETRIES, exit."
        exit 1
    fi

    sleep 1

    displayPortNum=$( getFreePortNumber )
    echo "running $PROFILE_NAME with displayPortNum=$displayPortNum"
    if [ -n "$DEBUG" ]
    then
        ./grpc-go -i $INPUTSRC -u 127.0.0.1:$GRPC_PORT -h 0.0.0.0:$displayPortNum
    else
        ./grpc-go -i $INPUTSRC -u 127.0.0.1:$GRPC_PORT -h 0.0.0.0:$displayPortNum 2>&1  | tee >/tmp/results/r"$cid"_"$PROFILE_NAME".jsonl >(stdbuf -oL sed -n -e 's/^.*fps: //p' | stdbuf -oL cut -d , -f 1 > /tmp/results/pipeline"$cid"_"$PROFILE_NAME".log)
    fi
    retry_cnt=$(( retry_cnt + 1 ))
    hasError5=$(grep "ERROR(5)-" /tmp/results/r"$cid"_"$PROFILE_NAME".jsonl)
done

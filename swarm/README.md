# Deploy using docker swarm

This document guides you to deploy a cluster using docker swarm with physical nodes.

## Pre-requisite 

- Ubuntu 22.04
- [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)

## Build docker images on all nodes

Ensure that **grpc_python:dev** and **cam-sim:dev** docker images are built on every node.
You can transfer the docker images to the other nodes or you can clone and run this command on every node to ensure the images are available.

Run these commands at the project's root level.

```
make build-grpc_python
```

Download models and media files:

```
make prepare-inputs
```

## Setting up Docker swarm

Start swarm mode

```
docker swarm init
```

### Join other worker nodes

For now, only the manager node will be available, if you want to join other nodes, execute the following command:

```
docker swarm join-token worker
```

This command will output something like the following, including a docker swarm join command with the token:

```
To add a worker to this swarm, run the following command:

    docker swarm join --token [TOKEN] 192.168.99.100:2377
```

On the new machine that you want to add as a worker node, run the docker swarm join command that you copied from the manager node. It should look something like this:


```
docker swarm join --token [TOKEN] 192.168.99.100:2377
```

Make sure to replace the token and IP address with the actual values you got from your manager node. This command will connect the worker node to the Docker Swarm managed by the manager node.

### Verify the nodes

To confirm that the new node has successfully joined the swarm, go back to your manager node and run:

```
docker node ls
```

| ID              | HOSTNAME        | STATUS | AVAILABILITY | MANAGER STATUS | ENGINE VERSION |
|-----------------|-----------------|--------|--------------|----------------|----------------|
| 1216415 *       | manager         | Ready  | Active       | Leader         | 23.0.3         |
| 1231321         | worker1         | Ready  | Active       |                | 24.0.7         |


## Deploy the grpc_python use case

Before deploying the stack, modify the **swarm/docker-compose.yml** file to deploy OVMS on master node and grpc_python on the client node.

Under **OvmsClientGrpcPython** section, replace the hostname with the worker node as the step above:

```yaml
      placement:
        constraints:
          - node.hostname == worker1  
```

Do the same step under the **ovmsServer** section, and replace it with the manager node: 

```yaml
      placement:
        constraints:
          - node.hostname == manager  
```

With your Swarm initialized, deploy the stack using the following command:

```
docker stack deploy -c swarm/docker-compose.yml grpc
```

### Verify the deployment

Check the status of your stack deployment using the following command:

```
docker stack services grpc
```

Output:

| ID             | NAME                         | MODE        | REPLICAS | IMAGE                              | PORTS                      |
|----------------|------------------------------|-------------|----------|------------------------------------|----------------------------|
| sadfdsfdsfdd   | grpc_OvmsClientGrpcPython    | replicated  | 1/1      | grpc_python:dev                    |                            |
| sdfsdfsdfdsf   | grpc_camera-simulator        | replicated  | 1/1      | cam-sim:dev                        | *:8554->8554/tcp           |
| dsfewfdesfdf   | grpc_camera-simulator0       | replicated  | 1/1      | cam-sim:dev                        |                            |
| ia8pj20vq2hz   | grpc_ovmsServer              | replicated  | 1/1      | ovms-server:dev                    | *:9001-9002->9001-9002/tcp |

### Inspect ovmsServer and OvmsClient

To verify that OVMS Server and Client are running on different nodes, execute the following command:

```
docker service ps grpc_ovmsServer
```

Output:

| ID         | NAME                      | IMAGE            | NODE            | DESIRED STATE | CURRENT STATE            | ERROR | PORTS |
|------------|---------------------------|------------------|-----------------|---------------|--------------------------|-------|-------|
| 7u9vyb798s | grpc_ovmsServer.1         | ovms-server:dev  | manager         | Running       | Running 6 minutes ago    |       |       |

## Stop the swarm

```
docker stack rm grpc
```
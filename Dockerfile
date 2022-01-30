FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

WORKDIR /app
COPY . .

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends python3.8 graphviz

RUN python3 -m pip install --upgrade pip \
    && pip3 --disable-pip-version-check --no-cache-dir install -r requirements.txt

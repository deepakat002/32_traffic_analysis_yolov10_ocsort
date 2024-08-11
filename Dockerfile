FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get -y install libgl1 libglib2.0-0 python3 >/dev/null 2>&1
RUN apt-get install -y python3-pip git wget curl ffmpeg libsm6 libxext6 gcc unzip g++ apt-utils >/dev/null 2>&1
RUN apt-get install lsb-release curl gpg -y >/dev/null 2>&1

RUN mkdir /trafficyolo

WORKDIR /trafficyolo

COPY requirements.txt .
COPY pyscripts .
RUN ls -lh

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/THU-MIG/yolov10.git



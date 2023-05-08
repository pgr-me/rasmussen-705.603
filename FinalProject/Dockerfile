FROM pytorch/pytorch:latest

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install wget ffmpeg libsm6 libxext6 gcc libpq-dev -y

WORKDIR /work

COPY FinalProject/. /work
COPY data/FinalProject/raw /work/data/raw

RUN apt-get update &&\
    /opt/conda/bin/python -m pip install --upgrade pip &&\
    pip3 install -r requirements.txt

RUN mkdir -p /work/data/raw /work/data/processed /work/data/interim

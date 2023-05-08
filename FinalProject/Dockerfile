FROM pytorch/pytorch:latest

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install wget ffmpeg libsm6 libxext6 -y

WORKDIR /work

COPY FinalProject/. /work
COPY data/FinalProject/raw /work/data/raw
#RUN pip install -r requirements.txt

RUN mkdir -p /work/data/raw /work/data/processed /work/data/interim

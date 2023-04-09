FROM pytorch:latest

WORKDIR /rl

COPY Assignment10/. /rl
COPY data/Assignment10/. /rl/data

RUN apt update &&\
    apt install vim -y &&\
    pip3 install -r requirements.txt

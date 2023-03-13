FROM mongo:latest

WORKDIR /nosql/mongo

COPY Assignment7/MongoDB/. /nosql/mongo
COPY data/Assignment7/raw/.  /import

RUN apt update &&\
    apt install software-properties-common -y &&\
    apt install python3-pip -y &&\
    apt install vim -y &&\
    pip3 install -r requirements.txt &&\
    mkdir -p /data/db


FROM neo4j

WORKDIR /nosql

COPY Assignment7/Neo4J/. /nosql

RUN apt update &&\
    apt install software-properties-common -y &&\
    apt install python3-pip -y &&\
    pip3 install -r requirements.txt 

FROM mongo:latest

WORKDIR /nosql/mongo

COPY Assignment7/Neo4J/. /nosql/neo4j
COPY data/Assignment7/raw/.  /nosql/neo4j/import

RUN apt update &&\
    apt install software-properties-common -y &&\
    apt install python3-pip -y &&\
    apt install vim -y &&\
    pip3 install -r requirements.txt &&\
    mkdir -p /nosql/neo4j/data 


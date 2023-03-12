FROM neo4j

WORKDIR /nosql/neo4j

COPY Assignment7/Neo4J/. /nosql/neo4j
COPY data/Assignment7/raw/.  /nosql/neo4j/import

RUN apt update &&\
    apt install software-properties-common -y &&\
    apt install python3-pip -y &&\
    pip3 install -r requirements.txt &&\
    mkdir -p /nosql/neo4j/data

CMD ["python3", "neo4jdating.py"]

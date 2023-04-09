# Assignment 7: Neo4J and MongoDB

## Project description

This program provides code to analyze a speed dating dataset using 1) Neo4J and 2) MongoDB as two separate approaches to organize the data for use in a decision tree classifier to predict which dates end in matches..

## Assignment structure

The assignment is sub-divided into Neo4J and MongoDB folders. Each has its own set of scripts and Dockerfiles to reproduce the work in a container.
```
Assignment7
├── MongoDB
│   ├── Dockerfile
│   ├── mongodating.ipynb
│   ├── mongodating.py
│   ├── readme.md
│   ├── requirements.txt
│   └── utils.py
├── Neo4J
│   ├── Dockerfile
│   ├── cypher
│   │   ├── female_male_ratio.cypher
│   │   ├── match_frac.cypher
│   │   └── schema_viz.cypher
│   ├── neo4dating.py
│   ├── neo4jdating.ipynb
│   ├── readme.md
│   ├── requirements.txt
│   └── utils.py
└── readme.md
```

## Docker

Rather than clone the repository to run the code, the user can opt to pull Docker image for this assignment from [the DockerHub repo](https://hub.docker.com/repository/docker/pgrjhu/705.603/general).

Pull the image: 
* Neo4J: `$ docker pull pgrjhu/705.603:a7-neo4j`
* MongoDB: `$ docker pull pgrjhu/705.603:a7-mongodb`

Run the executable to reproduce the results: 
* MongoDB:
    ```
    docker run \
        -p 27017:27017 -p 8888:8888 \
        -dit \
        -v /nosql/mongo/data:/data/db \
        -v /nosql/mongo/import:/import \
        --name mymongo \
        pgrjhu/705.603:a7-mongodb
    ```
* Neo4J:
    ```
    docker run \
       -p 7474:7474 -p 7687:7687 -p 8888:8888 \
       -d \
       --restart unless-stopped \
       -v /nosql/neo4j/data:/data \
       -v /nosql/neo4j/import:/var/lib/neo4j/import \
       --name neo4j \
       --env NEO4J_dbms_connector_http_advertised__address="localhost:7474" \
       --env NEO4J_dbms_connector_bolt_advertised__address="localhost:7687" \
       --env NEO4J_dbms_connector_https_advertised__address="localhost:7473" \
       --env NEO4J_AUTH=none \
       pgrjhu/705.603:a7-neo4j
    ```
Additional instructions for running an image are provided in the [DockerHub repo README](https://hub.docker.com/repository/docker/pgrjhu/705.603/general).



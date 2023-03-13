# Assignment 7: Neo4J and MongoDB

## Project description

This folder provides code to analyze a speed dating dataset using MongoDB to organize the data for use in a decision tree classifier to predict which dates end in matches.

## Assignment structure

```
.
├── Dockerfile
├── mongodating.ipynb
├── mongodating.py
├── readme.md
├── requirements.txt
└── utils.py
```

## Docker

Rather than clone the repository to run the code, the user can opt to pull Docker image for this assignment from [the DockerHub repo](https://hub.docker.com/repository/docker/pgrjhu/705.603/general).

Pull the image: 
* Neo4J: `$ docker pull pgrjhu/705.603:a7-neo4j`
* MongoDB: `$ docker pull pgrjhu/705.603:a7-mongodb`

Run the executable to reproduce the results: 
```
docker run \
   -p 27017:27017 -p 8888:8888 \
   -dit \
   -v /nosql/mongo/data:/data/db \
   -v /nosql/mongo/import:/import \
   --name mymongo \
   pgrjhu/705.603:a7-mongodb
```

Additional instructions for running an image are provided in the [DockerHub repo README](https://hub.docker.com/repository/docker/pgrjhu/705.603/general).


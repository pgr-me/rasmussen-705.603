# Assignment 4: Neo4J

## Project description

This program builds a Neo4J container.

## Data organization

Inputs are downloaded into the `raw` data directory. The program saves outputs in the `processed` directory. This program is idempotent: the inputs are never overwritten; only new outputs are created.
```
.
├── work
│   ├── Assignment4.py
│   └── Assignment4.ipynb
├── data
│   ├── processed
│   └── raw
│       ├── Musical_Instruments_5.json
│       └── Musical_instruments_reviews.csv
└── readme.md
```

## Docke

Rather than clone the repository to run the code, the user can opt to pull Docker image for this assignment from [the DockerHub repo](https://hub.docker.com/repository/docker/pgrjhu/705.603/general).

Pull the image: `$ docker pull pgrjhu/705.603:a7-neo4j`

Run the executable to reproduce the results: 
```
docker run -p 7474:7474 -p 7687:7687 \
   -d --restart unless-stopped \
   -v /nosql/neo4j/data:/data \
   -v /nosql/neo4j/import:/import \
   --name neo4j \
   pgrjhu/705.603:a7-neo4j
```

Additional instructions for running an image are provided in the [DockerHub repo README](https://hub.docker.com/repository/docker/pgrjhu/705.603/general).

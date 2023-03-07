# Assignment 4: Neo4J

## Project description

This program builds a Neo4J container.

## Data organization

Inputs are downloaded into the `raw` data directory. The program saves outputs in the `processed` directory. This program is idempotent: the inputs are never overwritten; only new outputs are created.
```
.
├── nosql/neo4j
│   ├── Assignment7.py
│   ├── Assignment7.ipynb
│   └── Assignment7.ipynb
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
docker run -p 7474:7474 -p 7687:7687 -p 8888:8888 \
   -d --restart unless-stopped \
   -v /nosql/neo4j/data:/data \
   -v /nosql/neo4j/import:/import \
   --name neo4j \
   pgrjhu/705.603:a7-neo4j
```

Additional instructions for running an image are provided in the [DockerHub repo README](https://hub.docker.com/repository/docker/pgrjhu/705.603/general).


## Reproducing the output


```
LOAD CSV WITH HEADERS FROM "https://raw.githubusercontent.com/jhebelerDS/NoSqlExercise/main/data/SpeedDatingData.csv" AS row
WITH row WHERE
    NOT row.age IS null
    AND NOT row.iid IS null
    AND NOT row.pid IS null
    AND NOT row.age_o IS null
    AND NOT row.race IS null
    AND NOT row.race_o IS null
    AND NOT row.match IS null
    AND NOT row.int_corr IS null
    AND NOT row.gender IS null
    AND NOT row.samerace IS null
    AND NOT row.age_o IS null
MERGE(
    p1 :Person {id:row.iid,age:toInteger(row.age),
    race:toInteger(row.race) }
)
MERGE(
    p2: Person {id:row.pid, age:toInteger(row.age_o), race:toInteger(row.race_o)}
)
MERGE(
    (p1) - [:Date {match: toInteger(row.match),
    int_corr: row.int_corr,
    race_diff:toInteger(row.samerace),
    age_diff:abs(toInteger(row.age)- toInteger(row.age_o))}] -> (p2)
)
SET p1.gender = toInteger(row.gender)
```

Visualize the schema:
```
CALL db.schema.visualization()
```

## Jupyter

To start a Jupyter Lab session, do:
`$ jupyter lab --no-browser --ip=0.0.0.0 --allow-root`

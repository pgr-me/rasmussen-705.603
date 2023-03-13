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

Get counts of males and females (assuming males are 0 and females are 1):
```
MATCH (n:Person)
WHERE n.gender=0
WITH count(n) AS count
RETURN 'male' AS label, count
UNION ALL
MATCH (n:person)
WHERE n.gender=1
WITH count(n) as count
RETURN 'female' as label, count
```

Get the unique set of mutual matches among speed daters:
```
MATCH (p1:Person)-[:Date {match: 1}]->(p2:Person)
WHERE
p1.id < p2.id
RETURN DISTINCT p1.id, p2.id
ORDER BY p1.id, p2.id
```

## Jupyter

To start a Jupyter Lab session, do:
`$ jupyter lab --no-browser --ip=0.0.0.0 --allow-root`

# Assignment 4: Text preprocessing

## Project description

This program downloads and preprocesses Amazon musical instrument review text for subsequent use in natural language processing analysis. By default, the program processes the `summary`column of the downloaded CSV, followed by tokenization, stemming, and lemmatization. Since stemming and lemmatization are used in lieu of one another, the program generates two outputs: one that has been stemmed and the other which has been lemmatized. Final outputs that are printed to console are top words by TFIDF score averaged across all documents.

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

## DockerHub

Rather than clone the repository to run the code, the user can opt to pull Docker image for this assignment from [the DockerHub repo](https://hub.docker.com/repository/docker/pgrjhu/705.603/general).
* Pull the image: `$ docker pull pgrjhu/705.603:module4-1.1`
* Run the executable to reproduce the results: `$ docker run [image ID]`.

Additional instructions for running an image are provided in the [DockerHub repo README](https://hub.docker.com/repository/docker/pgrjhu/705.603/general).

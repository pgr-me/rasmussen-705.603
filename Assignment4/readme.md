# Assignment 2: Text preprocessing

## Project description

This program downloads and preprocesses Amazon musical instrument review text for subsequent use in natural language processing analysis. By default, the program processes the `summary`column of the downloaded CSV, followed by tokenization, stemming, and lemmatization. Since stemming and lemmatization are used in lieu of one another, the program generates two outputs: one that has been stemmed and the other which has been lemmatized. Final outputs that are printed to console are top words by TFIDF score averaged across all documents.

## Data organization

Inputs are downloaded into the `raw` data directory. The program saves outputs in the `processed` directory. This program is idempotent: the inputs are never overwritten; only new outputs are created.

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


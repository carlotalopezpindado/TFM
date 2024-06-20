# TFM
This repository contains all the code used for the development of my Master Thesis.
The project consists of the development of a RAG system with BOE documents and access limitations.

## Table of contents

- [**TFM**](#tfm)
  - [Table of contents](#table-of-contents)
  - [Overview](#overview)
  - [Usage](#usage)
  - [Code Structure](#code-structure)
  - [License](#license)

## Overview
![System's login page](https://github.com/carlotalopezpindado/TFM/blob/main/images/login.png?raw=true)
![System's chat page](https://github.com/carlotalopezpindado/TFM/blob/main/images/chatbot.png?raw=true)


## Usage
This project has been prepared to run through **Docker-compose**. It can be started with two different files, depending if the device you are using has GPUs available or not. 

The [``docker-compose.gpu``](https://github.com/carlotalopezpindado/TFM/blob/main/docker-compose.gpu.yml) file should be used if there's GPUs available.
The [``docker-compose.nogpu``](https://github.com/carlotalopezpindado/TFM/blob/main/docker-compose.nogpu.yml) file should be used if there isn't GPUs available.


In order to deploy the sistem it is first necessary to download the necessary files with the indexes. This files can be found [here](https://drive.google.com/drive/folders/12RtxFgoqzOgyhsKLi4ao9qXCSMrS2uwL?usp=drive_link), and would need to be placed in the folder indexing/indices for the system to work properly.

After this has been done, the project can be started following these comands:

```bash
docker-compose -f docker-compose.gpu.yml build
```

```bash
docker-compose -f docker-compose.gpu.yml up
```

## Code structure
```bash
TFM/
├── Dockerfile
├── LICENSE
├── README.md
├── app/
│   └── main.py
├── docker-compose.gpu.yml
├── docker-compose.nogpu.yml
├── config.ini
├── evaluation/
│   ├── bertscore/
│   │   ├── bert_scores_adm.csv
│   │   ├── bert_scores_edu.csv
│   │   ├── bert_scores_leg.csv
│   │   ├── bertscore.py
│   │   └── metrics.ipynb
│   ├── factscore/
│   │   ├── factscore.py
│   │   ├── fs_adm.csv
│   │   ├── fs_edu.csv
│   │   ├── fs_leg.csv
│   │   └── metrics.ipynb
│   ├── qa_generation/
│   │   ├── qa_adm.csv
│   │   ├── qa_edu.csv
│   │   ├── qa_generation.py
│   │   └── qa_leg.csv
│   └── rag_answers/
│       ├── rag_adm.parquet
│       ├── rag_answers.py
│       ├── rag_edu.parquet
│       └── rag_leg.parquet
├── indexing/
│   ├── indexes/
│   │   ├── adm/
│   │   ├── edu/
│   │   └── leg/
│   ├── indexing.py
│   └── pruebas_indices.ipynb
├── mysql_database/
│   ├── Dockerfile
│   └── init_db.sql
├── preprocessing/
│   ├── crawling/
│   │   ├── BOE.txt
│   │   └── crawling.py
│   ├── scrapping/
│   │   └── scrapping.py
│   └── zeroshot-classification/
│       └── zeroshot-classification.py
└── requirements.txt
```

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

#!/bin/bash

# Dragonball
# Define URLs
DRAGONBALL_URL1="https://raw.githubusercontent.com/OpenBMB/RAGEval/main/dragonball_dataset/dragonball_queries.jsonl"
DRAGONBALL_URL2="https://raw.githubusercontent.com/OpenBMB/RAGEval/main/dragonball_dataset/dragonball_docs.jsonl"

# Define output file names
DRAGONBALL_OUTPUT1="data/Dragonball/data/dragonball_queries.jsonl"
DRAGONBALL_OUTPUT2="data/Dragonball/data/dragonball_docs.jsonl"

# Download files
wget -O $DRAGONBALL_OUTPUT1 $DRAGONBALL_URL1
wget -O $DRAGONBALL_OUTPUT2 $DRAGONBALL_URL2

echo "Dragonball Download complete."

cd data/Dragonball/ || exit
python preprocess.py
cd ../.. || exit

echo "Dragonball preprocess complete. Running preprocess..."

# Define URLs
CLAPNQ_URL1="https://github.com/primeqa/clapnq/raw/main/retrieval/train/question_train_answerable.tsv"
CLAPNQ_URL2="https://github.com/primeqa/clapnq/raw/main/retrieval/train/question_train_unanswerable.tsv"

# Define output file names
CLAPNQ_OUTPUT1="data/CLAPNQ/data/question_train_answerable.tsv"
CLAPNQ_OUTPUT2="data/CLAPNQ/data/question_train_unanswerable.tsv"

# Download files
wget -O $CLAPNQ_OUTPUT1 $CLAPNQ_URL1
wget -O $CLAPNQ_OUTPUT2 $CLAPNQ_URL2

echo "CLAPnq Download complete."

cd data/CLAPNQ || exit
python preprocess.py
cd ../..

echo "CLAPnq preprocess complete. Running preprocess..."
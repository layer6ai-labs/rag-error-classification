#!/bin/bash

# Define URLs
URL1="https://raw.githubusercontent.com/OpenBMB/RAGEval/main/dragonball_dataset/dragonball_queries.jsonl"
URL2="https://raw.githubusercontent.com/OpenBMB/RAGEval/main/dragonball_dataset/dragonball_docs.jsonl"

# Define output file names
OUTPUT1="dragonball_queries.jsonl"
OUTPUT2="dragonball_docs.jsonl"

# Download files
wget -O $OUTPUT1 $URL1
wget -O $OUTPUT2 $URL2

echo "Download completed."
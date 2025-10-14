import json

import pandas as pd


def filter_english_entries(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            entry = json.loads(line)
            if entry.get("language") == "en":
                outfile.write(json.dumps(entry) + "\n")


# Filter English entries in dragonball_docs.jsonl
filter_english_entries("data/dragonball_docs.jsonl", "data/dragonball_docs_en.jsonl")

# Filter English entries in dragonball_queries.jsonl
filter_english_entries("data/dragonball_queries.jsonl", "data/dragonball_queries_en.jsonl")

# Preprocess Queries
with open("data/dragonball_queries_en.jsonl", "r", encoding="utf-8") as file:
    query_data = [json.loads(line) for line in file]

ambiguous_df = pd.read_csv('data/ambiguous_queries_by_GPT.csv')

data = []
for d in query_data:
    query = d["query"]
    output = d["ground_truth"]
    data_dict = {
        "id": query["query_id"],
        "input": query["content"],
        "output": {
            "answers": [output["content"]],
            "provenance": [{"id": output["doc_ids"], "text": output["references"]}],
            "keypoints": output["keypoints"],
        },
        "metadata": {"domain": d["domain"], "query_type": query["query_type"], "ambiguous": False},
    }
    if data_dict['id'] in ambiguous_df['id'].values:
        if query['content'] == ambiguous_df.loc[ambiguous_df['id']==data_dict['id']]['input'].values[0]:
            data_dict['metadata']['ambiguous'] = True
    data.append(data_dict)

with open("data/dragonball_queries_en_processed.jsonl", "w", encoding="utf-8") as outfile:
    for entry in data:
        json.dump(entry, outfile)
        outfile.write("\n")

# Preprocess Docs
with open("data/dragonball_docs_en.jsonl", "r", encoding="utf-8") as file:
    docs_data = [json.loads(line) for line in file]
corpus = []
for doc in docs_data:
    doc_data_dict = {
        "doc_id": doc["doc_id"],
        "text": doc["content"],
        "doc_title": next((value for key, value in doc.items() if key.endswith("_name")), None),
        "metadata": {"domain": doc["domain"]},
        "text_chunks": [{"para_id": 0, "text": doc["content"]}],
    }
    corpus.append(doc_data_dict)

with open("data/dragonball_corpus_en_processed.jsonl", "w", encoding="utf-8") as outfile:
    for entry in corpus:
        json.dump(entry, outfile)
        outfile.write("\n")

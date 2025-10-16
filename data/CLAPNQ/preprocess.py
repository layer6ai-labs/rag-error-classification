from datasets import load_dataset
import json
import pandas as pd

# Read the TSV file
question_train_answerable = pd.read_csv('data/question_train_answerable.tsv', sep='\t')
question_train_unanswerable = pd.read_csv('data/question_train_unanswerable.tsv', sep='\t')
df_passages = pd.read_csv("hf://datasets/PrimeQA/clapnq_passages/passages.tsv", sep="\t")

document_data_list = []
for _, row_passage in df_passages.iterrows():
    data_doc = {
        'doc_id': row_passage['id'],
        'doc_title': row_passage['title'],
        'text_chunks': [{"para_id": 0, 'text': row_passage['text']}],
        'text': row_passage['text']
    }
    document_data_list.append(data_doc)

# Save the document_data_list to a JSONL file
with open('data/clapnq_docs.jsonl', 'w') as jsonl_file:
    for item in document_data_list:
        jsonl_file.write(json.dumps(item) + '\n')

id = 1
question_data_list = []

for index, row in question_train_answerable.iterrows():
    # Create a dictionary for each row
    data = {
        'id': id,
        'input': row['question'],
        "output": {
            'answer': row['answers'],
            'provenance': [{'id': row['doc-id-list'], 'text':
                df_passages[df_passages['id'] == row['doc-id-list']]['text'].values[0]}],
        },
        'unanswerable': False,
        'original_nq_id': row['id'],
    }
    # Increment the id for the next row
    id += 1
    question_data_list.append(data)
    row_passage = df_passages[df_passages['id'] == row['doc-id-list']].to_dict('records')[0]

for index, row in question_train_unanswerable.iterrows():
    # Create a dictionary for each row
    data = {
        'id': id,
        'input': row['question'],
        "output": {
            'answer': None,
            'provenance': [{}],
        },
        'unanswerable': True,
        'original_nq_id': row['id'],
    }
    # Increment the id for the next row
    id += 1
    question_data_list.append(data)

# Save the question_data_list to a JSONL file
with open('data/clapnq_queries.jsonl', 'w') as jsonl_file:
    for item in question_data_list:
        jsonl_file.write(json.dumps(item) + '\n')

# CLAP NQ Dataset

CLAP NQ is a dataset created from a subset of the Natural Questions (NQ) dataset. NQ consists of approximately 380,000 examples. CLAP NQ focuses on questions that have a long answer but no short answer, excluding tables and lists. To increase the likelihood of longer answers, only passages with more than 5 sentences were considered. The annotated subset consists of approximately 12,000 examples.


## Download Data

The data for CLAP NQ can be downloaded from the following repository:  
[CLAP NQ Retrieval Train Data](https://github.com/primeqa/clapnq/tree/main/retrieval/train)
You can download the following files for training:

- [question_train_answerable.tsv](https://github.com/primeqa/clapnq/raw/main/retrieval/train/question_train_answerable.tsv): Save this file to `data/CLAPNQ/data/question_train_answerable.tsv`.
- [question_train_unanswerable.tsv](https://github.com/primeqa/clapnq/raw/main/retrieval/train/question_train_unanswerable.tsv): Save this file to `data/CLAPNQ/data/question_train_unanswerable.tsv`.

## Preprocess
Preprocessing code is available in [pre_process.py](pre_process.py). The notebook includes steps to filter English entries from the dataset and organize the data in the desired format


## Queries
Each line in the query file is a JSON object representing a query with the following fields:

- **id**: A unique identifier for the query.
- **input**: The query text.
- **output**: An object containing:
    - **answer**: The long-form answer to the query.
    - **provenance**: A list of objects, each containing:
        - **id**: The identifier of the supporting passage or document.
        - **text**: The text of the supporting passage or document.
- **unanswerable**: A boolean indicating whether the query is unanswerable.
- **original_nq_id**: The original identifier from the Natural Questions dataset.

```json
{
    "id": 1,
    "input": "who sang love the one you're with first",
    "output": {
        "answer": "`` Love the One You 're With '' is a song by folk rocker Stephen Stills . David Crosby and Graham Nash , Stills ' fellow members of Crosby , Stills & Nash , provide background vocals on the song . The song was also covered by a number of artists , including The Isley Brothers , Bucks Fizz , and Luther Vandross .",
        "provenance": [
            {
                "id": "813602675_557-1132",
                "text": "`` Love the One You 're With '' is a song by folk rocker Stephen Stills . It was released as the lead single from his debut self - titled studio album in November 1970 . The song , inspired by a remark Stills heard from musician Billy Preston , became his biggest hit single , peaking at No. 14 on the Billboard Hot 100 in early 1971 . David Crosby and Graham Nash , Stills ' fellow members of Crosby , Stills & Nash , provide background vocals on the song . The song was also covered by a number of artists , including The Isley Brothers , Bucks Fizz , and Luther Vandross ."
            }
        ]
    },
    "unanswerable": false,
    "original_nq_id": 8045984229282682032
}
```

## License

Please refer to the [CLAP NQ repository](https://github.com/primeqa/clapnq) for licensing information and terms of use.
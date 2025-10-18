from typing import Any, Dict, List

from local_datasets.document_datasets.document_dataset import DocumentDataset
from local_datasets.document_datasets.chunkers import (
    FixedLengthChunker,
    RecursiveChunker,
    FixedLengthChunkerChinese,
)

CHUNKERS = {
    "fixed length": FixedLengthChunker,
    "fixed length chinese": FixedLengthChunkerChinese,
    "recursive": RecursiveChunker,
}


def create_chunks(
    corpus: DocumentDataset,
    strategy: str = "fixed length",
    chunk_size: int = 1000,
    overlap_size: int = 25,
) -> List[Dict[str, Any]]:
    """
    Document contains a text and title field and we want to split the text into chunks.
    This is a simple example that splits the text into chunks of maximum <chunk_size> words based on
    a chunking <strategy> ("fixed length", "overlap" or "recursive"). Default = "fixed length".
    For "overlap", chunks overlap by <overlap_size> words.
    For "recursive", chunks contain full sentences and size is maximum <chunk_size> words.

    Corpus:{id, document_record}. A document record contains:
    1. id: unique identifier for the document
    2. title: title of the document
    3. text: text of the document

    Chunk:{id, chunk_record}. A chunk record contains:
    1. id: unique identifier for the chunk (doc_id_chunk_id) e.g. 1_0
    2. doc_id: unique identifier for the document
    3. chunk_id: unique identifier for the chunk
    4. title: title of the document
    5. text: text of the chunk
    """
    chunks: list[dict[str, Any]] = []

    for document in corpus.get_documents():

        if strategy in CHUNKERS.keys():
            chunks.extend(CHUNKERS[strategy](document, chunk_size, overlap_size).create_chunks())
        else:
            raise ValueError(f"Strategy {strategy} not implemented")

    return chunks

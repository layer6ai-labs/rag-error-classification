from abc import ABC, abstractmethod
from typing import List, Any, Dict

try:
    import nltk

    nltk.download("punkt_tab")
except ImportError:
    pass


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.
    """

    def __init__(self, document, chunk_size: int, overlap_size: int = 0):
        self.document = document
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    @abstractmethod
    def create_chunks(self) -> List[Dict[str, Any]]:
        """
        Abstract method to be implemented by subclasses for chunking text.
        """
        pass


class FixedLengthChunker(BaseChunker):
    """
    Chunker that splits text into overlapping fixed-size chunks of words.
    """

    def create_chunks(self) -> List[Dict[str, Any]]:

        chunks: List[Dict[str, Any]] = []

        doc_id = self.document["id"]
        title = self.document["title"]
        text = self.document["text"]
        words = text.split()
        start = 0
        chunk_num = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_id = "{}_{}".format(doc_id, chunk_num)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "para_id": chunk_num,
                    "title": title,
                    "text": " ".join(words[start:end]),
                }
            )
            start += self.chunk_size - self.overlap_size
            chunk_num += 1

        return chunks


class RecursiveChunker(BaseChunker):
    """
    Recursive Chunker that splits text by keeping full sentences.
    """

    def create_chunks(self) -> List[Dict[str, Any]]:

        chunks: List[Dict[str, Any]] = []

        doc_id = self.document["id"]
        title = self.document["title"]
        text = self.document["text"]
        sentences = nltk.sent_tokenize(text)

        current_chunk = []
        start = 0
        chunk_num = 0
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            if current_tokens + sentence_tokens <= self.chunk_size:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                end = start + current_tokens
                chunk_id = "{}_{}".format(doc_id, chunk_num)
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "para_id": chunk_num,
                        "title": title,
                        "text": " ".join(current_chunk),
                    }
                )
                start = end
                chunk_num += 1
                current_chunk = [sentence]
                current_tokens = sentence_tokens

        # The last chunk needs to be appending separately
        if current_chunk:
            end = start + current_tokens
            chunk_id = "{}_{}".format(doc_id, chunk_num)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "para_id": chunk_num,
                    "title": title,
                    "text": " ".join(current_chunk),
                }
            )

        return chunks


class FixedLengthChunkerChinese(BaseChunker):
    """
    Chunker that splits text into overlapping fixed-size chunks of words for Chinese version.
    """

    def create_chunks(self) -> List[Dict[str, Any]]:

        chunks: List[Dict[str, Any]] = []

        doc_id = self.document["id"]
        title = self.document["title"]
        text = self.document["text"]
        start = 0
        chunk_num = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_id = "{}_{}".format(doc_id, chunk_num)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "para_id": chunk_num,
                    "title": title,
                    "text": text[start:end],
                }
            )
            start += self.chunk_size - self.overlap_size
            chunk_num += 1

        return chunks

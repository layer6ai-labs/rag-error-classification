from typing import Any, Dict
from local_datasets.document_datasets.document_dataset import DocumentDataset
from local_datasets.document_datasets.utils import create_chunks


class DragonballDoc(DocumentDataset):
    def load_one_doc(
        self, line_json: Dict[str, Any], load_docs: bool = True, load_semantic_chunks: bool = False
    ):
        if load_docs:
            doc = {
                "id": line_json["doc_id"],
                "title": line_json.get("title", ""),
                "text": line_json["text"],
            }
            self.corpus[doc["id"]] = doc
        if load_semantic_chunks:
            # for now the chunk is the entire document
            for chunk in line_json.get("text_chunks", []):
                chunk_id = "{}_{}".format(line_json["doc_id"], chunk["para_id"])
                c = {
                    "chunk_id": chunk_id,
                    "doc_id": line_json["doc_id"],
                    "para_id": chunk["para_id"],
                    "title": line_json.get("title", ""),
                    "text": chunk["text"],
                }
                self.chunks.append(c)
        if self.chinese:
            self.chunks = create_chunks(self, strategy="fixed length chinese", chunk_size=100)
        else:
            self.chunks = create_chunks(self, strategy="fixed length", chunk_size=1000)

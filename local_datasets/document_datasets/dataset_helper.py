import pathlib

from local_datasets.document_datasets.dragonball import DragonballDoc
from local_datasets.document_datasets.clapnq import ClapnqDoc

DATASET_CLASSES = {
    'dragonball': DragonballDoc,
    'dragonball-chinese': DragonballDoc,
    'clapnq': ClapnqDoc,
}


def get_document_dataset(
    dataset_name: str,
    corpus_file: pathlib.Path,
    load_docs: bool = True,
    load_semantic_chunks: bool = False,
    chinese: bool = False,
):
    return DATASET_CLASSES[dataset_name](corpus_file, load_docs, load_semantic_chunks, chinese)

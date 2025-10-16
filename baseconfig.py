import dataclasses
import pathlib
from hydra.core.config_store import ConfigStore


@dataclasses.dataclass
class DataConfig:
    corpus_path: pathlib.Path
    """
    Path to the document dataset
    """

    corpus_name: str
    """
    The name of the document dataset
    """

    corpus_chunk_size: int
    """
    Chunk size during document chunking
    """
    
    corpus_overlap_size: int
    """
    Chunk overlap size during document chunking
    """

    load_semantic_chunks: bool
    chunking_strategy: str
    query_path: pathlib.Path
    """
    Path to query dataset
    """

    query_name: str


@dataclasses.dataclass
class EmbeddingConfig:
    model: str
    """
    Model name for embedding model
    """

    batch_size: int
    """
    Batch size for passing to the embedding model 
    """

    device: str
    """
    The device of the embedding model
    """


@dataclasses.dataclass
class RetrievalConfig:
    retrieve_topk: int
    """
    The Top K chunks to retrieve.
    """


@dataclasses.dataclass
class RerankConfig:
    rerank_topk: int
    """
    The Top K chunks to rerank.
    """

    model: str
    """
    The model ID for reranker model
    """

    tokenizer: str
    """
    The tokenizer for the reranker model. Should be the same with the model ID.
    """

    batch_size: int
    """
    The batch size for passing the chunks to the reranker model.
    """

    context_size: int
    """
    The context size of the reranker model.
    """

    device: str
    """
    The device of the reranker model.
    """


@dataclasses.dataclass
class GeneratorConfig:
    model: str
    """
    The model ID for the generator model.
    """
    tokenizer: str
    """
    The tokenizer for the generator model. Should be the same with the model ID.
    """

    batch_size: int
    """
    The batch size for passing the chunks to the generator model.
    """

    context_size: int
    """
    The context size of the generator model.
    """

    device: str
    """
    The device of the generator model.
    """


@dataclasses.dataclass
class LlmConfig:
    model: str
    device: str
    max_new_tokens: int
    delay: int


@dataclasses.dataclass
class PromptConstructorConfig:
    name: str
    kwargs: dict[str, str]


@dataclasses.dataclass
class ErrorClassificationConfig:
    model_name: str
    num_rounds: int
    concept_covered_threshold: float
    rerank_covered_threshold: float
    ground_truth_chunk_threshold: float
    ground_truth_rounds: int


@dataclasses.dataclass
class LocalRagConfig:
    data: DataConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig
    generator: GeneratorConfig
    error_classification: ErrorClassificationConfig


@dataclasses.dataclass
class MainConfig:
    artifact_run_name: str
    """
    The run name of the current run. The artifacts of the same run name will be shared.
    """

    rag: LocalRagConfig


cs = ConfigStore.instance()
cs.store(name="base_rag", node=MainConfig)

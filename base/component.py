import abc
import logging
import pathlib
import re
from typing import Any, Type, List, Dict

import numpy as np
import pandas as pd

from clients.local_embedding_client import LocalEmbeddingClient
from clients.local_generate_client import LocalGenerateClient
from base import artifact
from clients.local_rerank_client import LocalRerankClient
from clients.local_numpy_client import LocalNumpyVectorDBClient


class Component(abc.ABC):
    """
    A component of the running experiment. A component must only do one thing. The input and
    output of the component could be artifacts that could be saved and loaded as intermediate
    files.
    """

    def __init__(self, out_path: pathlib.Path, **kwargs):
        """
        Constructor

        Args:
            out_path:
                The output path of the component. The artifacts will be saved in the path.
                The path is not the file name.
            **kwargs:
                The input artifacts.
        """
        self.artifacts = self._parse_kwargs(kwargs)
        self.log = logging.getLogger(self.__class__.__name__)
        self.out_path = out_path

    @classmethod
    @abc.abstractmethod
    def get_required(cls) -> dict[str, Type[artifact.Artifact]]:
        """
        Returns a dict containing the names and the types of artifacts required for this component,
        i.e. input artifacts of the component.
        """

    @classmethod
    @abc.abstractmethod
    def get_output_artifacts_info(cls) -> list[tuple[str, Type[artifact.Artifact]]]:
        """
        Returns a list of tuples. Each tuple contains the file name and the type of the artifact
        outputted by this component.
        """

    def get_specific_output_artifacts_info(self) -> list[tuple[str, Type[artifact.Artifact]]]:
        return self.get_output_artifacts_info()

    @abc.abstractmethod
    def _run(self) -> list[Any]:
        """
        Run this component assuming the artifacts are loaded.
        """

    @classmethod
    def _parse_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, artifact.Artifact]:
        """
        Load the input artifacts if necessary.
        """
        required_artifact_types = cls.get_required()
        if len(kwargs) != len(required_artifact_types):
            raise ValueError("Number of kwargs exceed the required.")
        artifact_dict = {}
        query_length = None
        for name, required_artifact_type in required_artifact_types.items():
            value = kwargs.get(name)
            if value is None or not isinstance(value, artifact.Artifact):
                raise TypeError(
                    f"kwargs values for {name} must be an artifact type. Currently "
                    f"it is {type(value)}"
                )
            value = value.load()
            if isinstance(value, artifact.QueryArtifact):
                if query_length is None:
                    query_length = len(value)
                else:
                    if len(value) != query_length:
                        raise ValueError(
                            f"Query length of {name} does not match other the query"
                            f" length in the other artifacts."
                        )
            artifact_dict[name] = value
        return artifact_dict

    def run(self, save: bool = True, dry: bool = False) -> list[artifact.Artifact]:
        """
        Run the component.

        Args:
            save:
                If ``True``, the output artifact will also be saved to disk.
            dry:
                If ``True``, this component will not run. Instead, generate non-loaded output
                artifacts by this component based on the file names and the path given. Note that
                this only depends on the output path. See :py:func:`~Component.dry_run`

        Returns:
            A list of output artifacts.

        """
        if dry:
            artifact_list = []
            for info in self.get_specific_output_artifacts_info():
                path = self.out_path / info[0]
                clz = info[1]
                artifact_list.append(clz(path, artifact=None))
            return artifact_list

        self.log.info("Running...")
        outputs = self._run()
        self.log.info("Done.")
        artifacts_list = []
        for info, output in zip(self.get_specific_output_artifacts_info(), outputs):
            path = self.out_path / info[0]
            clz = info[1]

            out_artifact = clz(path, output)
            if save:
                out_artifact.save()
                self.log.info(f"Saved to {path}")
            artifacts_list.append(out_artifact)
        return artifacts_list

    @classmethod
    def dry_run(cls, out_path: pathlib.Path) -> list[artifact.Artifact]:
        """
        Generate non-loaded output artifacts by this component based on the file names and the
        path given.

        Args:
            out_path:
                The output path of where the artifacts should be loaded. Should not contain
                the file name.

        Returns:
            A list of (non-loaded) artifacts.

        """
        artifact_list = []
        for info in cls.get_output_artifacts_info():
            path = out_path / info[0]
            clz = info[1]
            artifact_list.append(clz(path, artifact=None))
        return artifact_list

    def get_object(self, name: str) -> Any:
        """
        Returns the underlying object from the artifact name.
        """
        fetched = self.artifacts.get(name)
        if fetched is None:
            raise ValueError(
                f"Artifact {name} not found. Must be in " f"{list(self.artifacts.keys())}"
            )
        return fetched.artifact


class ChunksEmbedder(Component):
    """
    A component that embed chunks

    Inputs:
        chunks:
            The document dataset artifact.
    Outputs:
        chunk_embeddings.npy:
            A numpy array of shape (num_chunks, num_hidden)
    """

    def __init__(
        self,
        embedding_client: LocalEmbeddingClient,
        emb_batch_size: int,
        out_path: pathlib.Path,
        **kwargs,
    ):
        """
        Constructor

        Args:
            embedding_client:
                The embedding client.
            emb_batch_size:
                The batch size for the embedding client to run.
        """
        super().__init__(out_path, **kwargs)
        self.emb_batch_size = emb_batch_size
        self.embedding_client = embedding_client

    @classmethod
    def get_required(cls) -> dict[str, Type[artifact.Artifact]]:
        return {
            "chunks": artifact.DocumentDatasetArtifact,
        }

    def _run(self) -> list[np.ndarray]:
        chunks = self.get_object("chunks")
        chunk_texts: list[str] = chunks.chunk_texts
        embeddings = self.embedding_client.encode(chunk_texts, self.emb_batch_size).numpy()
        return [embeddings]

    @classmethod
    def get_output_artifacts_info(cls) -> list[tuple[str, Type[artifact.Artifact]]]:
        return [("chunk_embeddings.npy", artifact.NumpyArtifact)]


class QueryEmbedder(Component):
    """
    A component that embed queries

    Inputs:
        queries:
            The query dataset artifact.
    Outputs:
        query_embeddings.npy:
            A numpy array of shape (num_queries, num_hidden)
    """

    def __init__(
        self,
        embedding_client: LocalEmbeddingClient,
        emb_batch_size: int,
        out_path: pathlib.Path,
        **kwargs,
    ):
        """
        Constructor

        Args:
            embedding_client:
                The embedding client.
            emb_batch_size:
                The batch size for the embedding client to run.
        """
        super().__init__(out_path, **kwargs)
        self.embedding_client = embedding_client
        self.emb_batch_size = emb_batch_size

    @classmethod
    def get_required(cls) -> dict[str, Type[artifact.Artifact]]:
        return {
            "queries": artifact.QueryDatasetArtifact,
        }

    def _run(self) -> list[np.ndarray]:
        queries = self.get_object("queries")
        query_inputs = queries.query_text
        embeddings = self.embedding_client.encode(query_inputs, self.emb_batch_size).numpy()
        return [embeddings]

    @classmethod
    def get_output_artifacts_info(cls) -> list[tuple[str, Type[artifact.Artifact]]]:
        return [("query_emebddings.npy", artifact.NumpyQueryArtifact)]


class Searcher(Component):
    """
    A component that search the chunk embeddings for each query embedding.

    Inputs:
        chunks:
            The document dataset artifact.
        chunk_embeddings:
            A numpy array of shape (num_chunks, num_hidden)
        query_embeddings:
            A numpy array of shape (num_queries, num_hidden)
    Outputs:
        search_result_ids.npy:
            A numpy array of shape (num_queries, retrieve_topk) indicating the chunk ids of the retrieved
            chunks
        search_result_chunks.npy:
            A numpy array of shape (num_queries, retrieve_topk) indicating the retrieved chunks. The dtype
            will be the chunk type (for example dict[str, Any])
        search_result_distances.npy
            A numpy array of shape (num_queries, retrieve_topk) indicating the distances of retrieved
            chunks to the query.
    """

    def __init__(self, retrieve_topk: int, out_path: pathlib.Path, **kwargs):
        """
        Constructor

        Args:
            retreive_topk:
                The top k search results to return per query.
        """
        super().__init__(out_path, **kwargs)
        self.vector_db_client = LocalNumpyVectorDBClient()
        self.retrieve_topk = retrieve_topk

    @classmethod
    def get_required(cls) -> dict[str, Type[artifact.Artifact]]:
        return {
            "chunks": artifact.DocumentDatasetArtifact,
            "chunk_embeddings": artifact.NumpyArtifact,
            "query_embeddings": artifact.NumpyQueryArtifact,
        }

    def _run(self) -> list[Any]:
        chunks = self.get_object("chunks")
        chunk_embeddings = self.get_object("chunk_embeddings")
        query_embeddings = self.get_object("query_embeddings")
        ids = chunks.chunk_ids
        self.vector_db_client.add(vectors=chunk_embeddings, ids=ids, payloads=chunks.chunks)
        ids, retrieved_chunks, distances = self.vector_db_client.search(
            query_embeddings, self.retrieve_topk
        )
        return [ids, retrieved_chunks, distances]

    @classmethod
    def get_output_artifacts_info(cls) -> list[tuple[str, Type[artifact.Artifact]]]:
        return [
            ("search_result_ids.npy", artifact.NumpyQueryArtifact),
            ("search_result_chunks.npy", artifact.NumpyQueryArtifact),
            ("search_result_distances.npy", artifact.NumpyQueryArtifact),
        ]


class Reranker(Component):
    """
    A component that rerank the search results.

    Inputs:
        retrieved_chunks:
            A numpy array of shape (num_queries, retrieve_topk) indicating the retrieved chunks. The dtype
            will be the chunk type (for example dict[str, Any])
        queries:
            The query dataset artifact.
    Outputs:
        rerank_order.npy:
            A numpy array of shape (num_queries, retrieve_topk) (so far) for the reranked indices of the
            top chunks. These are not the chunk indices, but the indices within
            {0, 1, .., retreive_topk - 1}.
        rerank_context.npy:
            A numpy array of shape (num_queries, rerank_topk) indicating the reranked chunks. The dtype
            will be the chunk type (for example dict[str, Any
    """

    def __init__(
        self,
        rerank_client: LocalRerankClient,
        out_path: pathlib.Path,
        rerank_batch_size: int = 1,
        num_passes: int = 1,
        window_size: int = 1000,
        step: int = 1000,
        rerank_topk: int = 3,
        **kwargs,
    ):
        """
        Constructor

        Args:
            rerank_client:
                The rerank client.
            num_passes:
                The number of passes of the reranker
            window_size:
                The window size of the reranker
            step:
                The step of the reranker
        """
        super().__init__(out_path, **kwargs)
        self.rerank_client = rerank_client
        self.rerank_batch_size = rerank_batch_size
        self.num_passes = num_passes
        self.window_size = window_size
        self.step = step
        self.rerank_topk = rerank_topk

    @classmethod
    def get_required(cls) -> dict[str, Type[artifact.Artifact]]:
        return {
            "retrieved_chunks": artifact.NumpyQueryArtifact,
            "queries": artifact.QueryDatasetArtifact,
        }

    @classmethod
    def get_output_artifacts_info(cls) -> list[tuple[str, Type[artifact.Artifact]]]:
        return [
            ("rerank_order.npy", artifact.NumpyQueryArtifact),
            ("rerank_context.npy", artifact.NumpyQueryArtifact),
        ]

    def _run(self) -> list[np.ndarray]:
        retrieved_chunks = self.get_object("retrieved_chunks")
        queries = self.get_object("queries").query_text
        rerank_order = self.rerank_client.rerank_batch(
            queries=queries,
            chunks=[[c["text"] for c in retrieved_chunk] for retrieved_chunk in retrieved_chunks],
            batch_size=self.rerank_batch_size,
            num_passes=self.num_passes,
            window_size=self.window_size,
            step=self.step,
        )
        rerank_context = np.take_along_axis(
            retrieved_chunks, rerank_order[:, : self.rerank_topk], axis=1
        )
        return [rerank_order, rerank_context]


class Generator(Component):
    """
    A component that generate the final output of the RAG given the reranked order and the queries.

    Inputs:
        rerank_context:
            A numpy array of shape (num_queries, rerank_topk) indicating the reranked chunks. The dtype
            will be the chunk type (for example dict[str, Any])
        queries:
            The query dataset artifact.
    Outputs:
        generated_text.csv:
            A pandas dataframe containing one column "generated_text" containing the generated
            texts.
        generated_logits.npy:
            A numpy array of shape (num_queries, output_token_length) containing the
            generated logits of the generated text. Note that output token length is the max length
            of the output tokens within a batch. The logits will be nan after (not at) the first
            eos token. If no logits are returned, it will be a numpy array shape (num_queries, 0).
    """

    def __init__(
        self,
        generate_client: LocalGenerateClient,
        out_path: pathlib.Path,
        generator_batch_size: int = 1,
        return_logits: bool = True,
        **kwargs,
    ):
        """
        Constructor

        Args:
            generate_client:
                The generator client.
            num_rerank_chunks:
                The number of reranked chunks to be fed into the generator.
            return_logits:
                a bool indicating whether the logits should be returned.
        """
        super().__init__(out_path, **kwargs)
        self.generate_client = generate_client
        self.generator_batch_size = generator_batch_size
        self.return_logits = return_logits

    @classmethod
    def get_required(cls) -> dict[str, Type[artifact.Artifact]]:
        return {
            "rerank_context": artifact.NumpyQueryArtifact,
            "queries": artifact.QueryDatasetArtifact,
        }

    @classmethod
    def get_output_artifacts_info(cls) -> list[tuple[str, Type[artifact.Artifact]]]:
        return [
            ("generated_text.csv", artifact.PandasQueryArtifact),
            ("generated_logits.npy", artifact.NumpyQueryArtifact),
        ]

    def _run(self) -> list[Any]:
        rerank_context = self.get_object("rerank_context")
        query_inputs = self.get_object("queries").query_text

        generated_text, generated_logits = self.generate_client.generate_batch(
            query_inputs,
            rerank_context,
            batch_size=self.generator_batch_size,
            return_logits=self.return_logits,
        )
        generated_text = pd.DataFrame(generated_text, columns=["generated_text"])
        return [generated_text, generated_logits]


class CitationExtractor(Component):
    """
    A component that parse the generated text and split it to prediction text and citations.

    Inputs:
        generated_text:
            A pandas dataframe containing one column "generated_text" containing the generated
            texts.
        queries:
            The query dataset artifact.

    Outputs:
        processed_output.pkl:
            A list of length number of queries, with each element being a dict, containing
            "id" - Query ID; "prediction_text" - The prediction text; "citations" - a list
            of strings containing the cited chunk IDs.

    """

    _CITIATION_PATTERN = re.compile(r"^(.*)\[(.*?)\](.*)$")

    @classmethod
    def get_required(cls) -> dict[str, Type[artifact.Artifact]]:
        return {
            "generated_text": artifact.PandasArtifact,
            "queries": artifact.QueryDatasetArtifact,
        }

    @classmethod
    def get_output_artifacts_info(cls) -> list[tuple[str, Type[artifact.Artifact]]]:
        return [("processed_output.pkl", artifact.PickleArtifact[List[Dict[str, Any]]])]

    def _run(self) -> list[Any]:
        queries = self.get_object("queries").queries
        text_list: list[str] = self.get_object("generated_text").iloc[:, 0].tolist()
        out_list = []
        for query, text in zip(queries, text_list):
            extracted = self._CITIATION_PATTERN.match(text)
            citations = (
                set() if extracted is None else {x.strip() for x in extracted.group(2).split(",")}
            )
            prediction_text = re.sub(
                self._CITIATION_PATTERN, r"\1\3", text.lstrip("Answer:")
            ).strip()
            out_list.append(
                {
                    "citations": citations,
                    "prediction_text": prediction_text,
                    "id": query["query_id"],
                }
            )
        return [out_list]

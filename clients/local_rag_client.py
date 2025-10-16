from __future__ import annotations

import logging
import pathlib

from autoeval.errorclassification.rulebased_error_classification import RuleBasedErrorClassification
from autoeval.errorclassification.conceptvalidator import ConceptValidator
from autoeval.errorclassification.ground_truth_chunks import GroundTruthChunks
from autoeval.llm_evaluator.model import LlmEvaluator
from baseconfig import LocalRagConfig
from clients.local_embedding_client import LocalEmbeddingClient
from clients.local_generate_client import LocalGenerateClient
from base import component
from base.artifact import QueryDatasetArtifact, DocumentDatasetArtifact
from base.rag_results import QueryResults
from clients.local_rerank_client import LocalRerankClient


class LocalRagClient:
    """
    The "pipeline" of the RAG system. This file contains the running logic of a RAG system.
    Previously it was part of bottlenose_test_local.py. Now the file was split into the
    argparse part, and the logic part. The logic part will be included in this file.
    """

    def __init__(self, config: LocalRagConfig, out_path: pathlib.Path):
        """
        Constructor

        Args:
            config:
                The rag config. See `:py:obj:LocalRagConfig`
            out_path:
                The output path indicating where the artifacts should be saved.
        """
        self.log = logging.getLogger(LocalRagClient.__name__)
        self.config = config
        self.out_path = out_path
        self.out_path.mkdir(exist_ok=True, parents=True)

    def run(
        self,
        chunk_sample: int | None = None,
        query_sample: int | None = None,
    ) -> QueryResults:
        # Document Dataset
        chunks = DocumentDatasetArtifact.from_config(self.config)
        if chunk_sample is not None:
            chunks = chunks[:chunk_sample]

        # Query dataset
        queries = QueryDatasetArtifact.from_config(self.config)
        if query_sample is not None:
            queries = queries[:query_sample]

        # Initiate clients
        embedding_client = LocalEmbeddingClient(self.config.embedding)
        rerank_client = LocalRerankClient(self.config.rerank)
        generate_client = LocalGenerateClient(self.config.generator)

        # Embedding Chunks
        chunk_embedder = component.ChunksEmbedder(
            embedding_client,
            self.config.embedding.batch_size,
            out_path=self.out_path,
            chunks=chunks,
        )
        (chunk_embeddings,) = chunk_embedder.run(save=True, dry=False)
        # <-- Second way to dry run, without creating the ChunksEmbedder instance.
        # (chunk_embeddings,) = component.ChunksEmbedder.dry_run(self.out_path)

        # Embedding Queries
        query_embedder = component.QueryEmbedder(
            embedding_client,
            self.config.embedding.batch_size,
            out_path=self.out_path,
            queries=queries,
        )
        (query_embeddings,) = query_embedder.run(save=True, dry=False)
        # (query_embeddings,) = component.QueryEmbedder.dry_run(self.out_path)

        # Vector DB Search
        searcher = component.Searcher(
            retrieve_topk=self.config.retrieval.retrieve_topk,
            out_path=self.out_path,
            chunks=chunks,
            query_embeddings=query_embeddings,
            chunk_embeddings=chunk_embeddings,
        )
        ids, retrieved_chunks, distances = searcher.run(save=True, dry=False)
        # ids, retrieved_chunks, distances = component.Searcher.dry_run(self.out_path)

        # Rerank
        reranker = component.Reranker(
            rerank_client=rerank_client,
            rerank_batch_size=self.config.rerank.batch_size,
            out_path=self.out_path,
            retrieved_chunks=retrieved_chunks,
            queries=queries,
            rerank_topk=self.config.rerank.rerank_topk,
        )
        rerank_order, rerank_context = reranker.run(save=True, dry=False)
        # rerank_order, rerank_context = component.Reranker.dry_run(self.out_path)

        # Generate
        generate = component.Generator(
            generate_client=generate_client,
            generator_batch_size=self.config.generator.batch_size,
            out_path=self.out_path,
            rerank_context=rerank_context,
            queries=queries,
            return_logits=True,
        )
        generated_text, generated_logits = generate.run(save=True, dry=False)
        # generated_text, generated_logits = component.Generator.dry_run(self.out_path)

        # print results
        query_results = QueryResults(queries=queries)
        query_results.add_query_artifacts("ID", ids)
        query_results.add_query_artifacts("Retrieved", retrieved_chunks)
        query_results.add_query_artifacts("Distance", distances)
        query_results.add_query_artifacts("Rerank Order", rerank_order)
        query_results.add_query_artifacts("Generated Text", generated_text)
        # query_results.add_query_artifacts("Generate Logit", generated_logits)
        query_results.print_result()
        return query_results

    def evaluate(self):
        generated_text, generated_logits = component.Generator.dry_run(self.out_path)
        queries = QueryDatasetArtifact.from_config(self.config)
        if len(generated_text.load().artifact) < len(queries.load().artifact):
            queries = queries[: len(generated_text.artifact)]

        citation_extractor = component.CitationExtractor(
            self.out_path,
            queries=queries,
            generated_text=generated_text,
        )
        (processed_output,) = citation_extractor.run(save=True, dry=False)

        llm_evaluator = LlmEvaluator(
            self.out_path,
            queries=queries,
            processed_output=processed_output,
        )
        (eval_result,) = llm_evaluator.run(save=True, dry=False)

    def classify_errors(self):
        chunks = DocumentDatasetArtifact.from_config(self.config)
        queries = QueryDatasetArtifact.from_config(self.config)
        (processed_output,) = component.CitationExtractor.dry_run(self.out_path)
        _, retrieved_chunks, _ = component.Searcher.dry_run(self.out_path)
        (llmeval_results,) = LlmEvaluator.dry_run(self.out_path)
        rerank_order, _ = component.Reranker.dry_run(self.out_path)

        ground_truth_chunks_comp = GroundTruthChunks(
            out_path=self.out_path,
            openai_model_name=self.config.error_classification.model_name,
            num_rounds=self.config.error_classification.ground_truth_rounds,
            queries=queries,
            chunks=chunks,
            llmeval_results=llmeval_results,
        )
        (ground_truth_chunks,) = ground_truth_chunks_comp.run(save=True, dry=False)

        concept_validator = ConceptValidator(
            out_path=self.out_path,
            openai_model_name=self.config.error_classification.model_name,
            ground_truth_threshold=self.config.error_classification.ground_truth_chunk_threshold,
            queries=queries,
            chunks=chunks,
            llmeval_results=llmeval_results,
            ground_truth_chunks=ground_truth_chunks,
        )
        (validated_concepts,) = concept_validator.run(save=True, dry=False)

        # noinspection PyTypeChecker
        error_classification = RuleBasedErrorClassification.from_config(
            out_path=self.out_path,
            config=self.config,
            queries=queries,
            chunks=chunks,
            processed_output=processed_output,
            reranked_order=rerank_order,
            retrieved_chunks=retrieved_chunks,
            llmeval_results=llmeval_results,
            ground_truth_chunks=ground_truth_chunks,
            validated_concepts=validated_concepts,
        )
        (errors,) = error_classification.run(save=True, dry=False)

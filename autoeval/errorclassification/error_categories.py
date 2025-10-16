import dataclasses
import enum


class RagFailureStage(str, enum.Enum):
    DOCUMENT_PREPROCESSING = "Chunking"
    RETRIEVAL_FAILURES = "Retrieval"
    RERANKING_FAILURES = "Reranking"
    ANSWER_GENERATION = "Generation"
    UNKNOWN = "unknown"

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


@dataclasses.dataclass
class ErrorCategory:
    name: str
    description: str
    stage: RagFailureStage

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, ErrorCategory):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


OVERCHUNKING = ErrorCategory(
    "Overchunking",
    "Document is splitted into excessively small chunks, causing important context to be lost. "
    "Individual chunks appear incomplete or ambiguous.",
    RagFailureStage.DOCUMENT_PREPROCESSING,
)
UNDERCHUNKING = ErrorCategory(
    "Underchunking",
    "Chunks are too large, covering multiple topics with mixed content. "
    "Individual chunks can be confusing and crucial information is diluted.",
    RagFailureStage.DOCUMENT_PREPROCESSING,
)
CONTEXT_MISMATCH = ErrorCategory(
    "Context Mismatch",
    "Chunks are splitted at arbitrary boundaries, disrupting the logical structure of the document. "
    "Key contextual links are seperated from the information they link to",
    RagFailureStage.DOCUMENT_PREPROCESSING,
)
MISSED_RETRIEVAL = ErrorCategory(
    "Missed Retrieval",
    "Retrieved chunks does not contain the relevant information to"
    "answer the query from the ground truth documents",
    RagFailureStage.RETRIEVAL_FAILURES,
)
LOW_RELEVANCE = ErrorCategory(
    "Low Relevance",
    "Retrieved chunks are only loosely related to the query",
    RagFailureStage.RETRIEVAL_FAILURES,
)
SEMANTIC_DRIFT = ErrorCategory(
    "Semantic Drift",
    "Retrieved chunks appear to match keywords but do not align with the query's intent",
    RagFailureStage.RETRIEVAL_FAILURES,
)
LOW_RECALL = ErrorCategory(
    "Low Recall",
    "Necessary chunks are retrieved but reranked too low and not forwarded to the generator",
    RagFailureStage.RERANKING_FAILURES,
)
LOW_PRECISION = ErrorCategory(
    "Low Precision",
    "Irrelevant chunks are reranked highly and forwarded to the generator, with the importance"
    "over the truly relevant chunks",
    RagFailureStage.RERANKING_FAILURES,
)
ABSTENTION_FAILURE = ErrorCategory(
    "Abstention Failure",
    "The model should have abstained but provided an incorrect answer",
    RagFailureStage.ANSWER_GENERATION,
)
FABRICATED_CONTENT = ErrorCategory(
    "Fabricated Content",
    "The response includes information not present in the source document chunks and is unverifiable",
    RagFailureStage.ANSWER_GENERATION,
)
PARAMETRIC_OVERRELIANCE = ErrorCategory(
    "Parametric Overreliance",
    "The response depends on the LLM's internal knowledge rather than the source document chunks",
    RagFailureStage.ANSWER_GENERATION,
)
INCOMPLETE_ANSWER = ErrorCategory(
    "Incomplete Answer",
    "The response is correct but missing critical details",
    RagFailureStage.ANSWER_GENERATION,
)
MISINTERPRETATION = ErrorCategory(
    "Misinterpretation",
    "The generator misuses or misrepresents the source document chunks",
    RagFailureStage.ANSWER_GENERATION,
)
CONTEXTUAL_MISALIGNMENT = ErrorCategory(
    "Contextual Misalignment",
    "The response is correct but does not address the query",
    RagFailureStage.ANSWER_GENERATION,
)
CHRONOLOGICAL_INCONSISTENCY = ErrorCategory(
    "Chronological Inconsistency",
    "The response presents events or facts in the wrong temporal order, or confuses past, present,"
    " or future timelines",
    RagFailureStage.ANSWER_GENERATION,
)
NUMERICAL_ERROR = ErrorCategory(
    "Numerical Error",
    "The response includes incorrect calculations, quantities, or misrepresents numerical data"
    " from the retrieved documents",
    RagFailureStage.ANSWER_GENERATION,
)

NA = ErrorCategory(
    "NA",
    "No Error",
    RagFailureStage.UNKNOWN,
)

ERROR_CATEGORIES = [
    OVERCHUNKING,
    UNDERCHUNKING,
    CONTEXT_MISMATCH,
    MISSED_RETRIEVAL,
    LOW_RELEVANCE,
    SEMANTIC_DRIFT,
    LOW_RECALL,
    LOW_PRECISION,
    ABSTENTION_FAILURE,
    FABRICATED_CONTENT,
    PARAMETRIC_OVERRELIANCE,
    INCOMPLETE_ANSWER,
    MISINTERPRETATION,
    CONTEXTUAL_MISALIGNMENT,
    CHRONOLOGICAL_INCONSISTENCY,
    NUMERICAL_ERROR,
]

# from typing import TypedDict, Optional, Annotated
# from operator import add

# class GenerationResponse(TypedDict):
#     instances: list[str]


# class FluencyResponse(TypedDict):
#     fluency_score: float
#     errors: dict[str, str]
#     summary: str


# class NaturalnessResponse(TypedDict):
#     naturalness_score: float
#     observations: dict[str, str]
#     summary: str


# class CSRatioResponse(TypedDict):
#     ratio_score: float
#     computed_ratio: str
#     notes: str

# class CSRatioResponsePerInstance(TypedDict):
#     """Response for a single sentence's CS ratio evaluation (from batch call)"""
#     ratio_score: float
#     computed_ratio: str
#     notes: str


# class SocialCulturalResponse(TypedDict):
#     socio_cultural_score: float
#     issues: str
#     summary: str

# class AgentRunningState(TypedDict, total=False):
#     topic: str
#     tense: str
#     perspective: str
#     cs_ratio: str
#     gender: str
#     age: str
#     education_level: str
#     first_language: str
#     second_language: str
#     conversation_type: str
#     cs_function: str
#     cs_type: str
#     news_article: Optional[str] = None
#     response: str
#     data_generation_result: list[str]
#     cs_ratio_results_per_instances: list[CSRatioResponsePerInstance]  # batch eval results

#     news_generation_result: list[str]
#     fluency_result: FluencyResponse
#     naturalness_result: NaturalnessResponse
#     social_cultural_result: SocialCulturalResponse

#     summary: str
#     score: float

#     refine_count: Annotated[int, add]

#     news_hash: set
#     news_dict: dict



from typing import TypedDict, Optional, Literal, Annotated, Any, List, Dict, Set, Union
from operator import add

# --- keep these EXACTLY as your original ---
class GenerationResponse(TypedDict):
    instances: list[str]

class FluencyResponse(TypedDict):
    fluency_score: float
    errors: dict[str, str]
    summary: str

class NaturalnessResponse(TypedDict):
    naturalness_score: float
    observations: dict[str, str]
    summary: str

class SocialCulturalResponse(TypedDict):
    socio_cultural_score: float
    issues: str
    summary: str

class CSRatioResponse(TypedDict):
    ratio_score: float
    computed_ratio: str
    notes: str


# --- add only NEW task-aware pieces ---
class NamedEntity(TypedDict, total=False):
    """One entity the NER validator reports as EVIDENCE for its verdict."""
    text: str          # exact span as it appears in the sentence
    type: str          # PER | ORG | LOC | ...


class TaskValidationResult(TypedDict, total=False):
    passed: bool
    confidence: float
    notes: str
    llm_notes: str
    deterministic_notes: str
    predicted_label: Optional[str]
    errors: list[str]
    # NER only: the entities the validator actually found. The verdict is RECOMPUTED in code
    # from this evidence (types present, counts, script, nesting) instead of trusting `passed`,
    # which was observed to be self-contradictory (passed=True alongside a blocking error).
    # Optional (total=False) so the topic/sentiment validators are unaffected.
    entities: list[NamedEntity]


class SentenceRecord(TypedDict, total=False):
    """Canonical per-sentence record combining text, all metrics, and lifecycle state.

    Each metric field holds the full typed response object (not just the score float),
    so all errors, notes, summaries, and computed values are preserved per sentence.
    Built in SummarizeResult and used as canonical runtime source where available.
    status values: 'pass' | 'fail' | 'refined_pass' | 'budget_exhausted'
    """
    index: int
    text: str
    fluency: FluencyResponse                    # full FluencyResponse per sentence
    naturalness: NaturalnessResponse            # full NaturalnessResponse per sentence
    cs_ratio: CSRatioResponse                   # full CSRatioResponse per sentence
    socio_cultural: SocialCulturalResponse      # full SocialCulturalResponse per sentence
    weighted_score: float
    refine_count: int
    status: str
    task_passed: Optional[bool]
    task_validation: TaskValidationResult      # full per-instance task validation object

class BaseState(TypedDict, total=False):
    task: Literal["topic", "sentiment", "ner"]

    # task-specific payload keys (must exist in base state so LangGraph keeps them)
    label: str
    task_constraints: Dict[str, Any]
    annotations: list["NERSpan"]

    # existing scenario keys
    topic: str
    tense: str
    perspective: str
    cs_ratio: str
    gender: str
    age: str
    education_level: str
    first_language: str
    second_language: str
    conversation_type: str
    cs_function: str
    cs_type: str
    use_tools: bool

    # generation outputs
    text: Optional[str]            # new canonical field (recommended)
    data_generation_result: list[str]
    response: str

    # quality results (unchanged types)
    task_validation_result: TaskValidationResult
    task_validation_results_per_instances: List[TaskValidationResult]
    fluency_result: FluencyResponse
    fluency_results_per_instances: List[FluencyResponse]
    naturalness_result: NaturalnessResponse
    naturalness_results_per_instances: List[NaturalnessResponse]
    cs_ratio_result: CSRatioResponse
    cs_ratio_results_per_instances: List[CSRatioResponse]
    social_cultural_result: SocialCulturalResponse
    social_cultural_results_per_instances: List[SocialCulturalResponse]

    summary: str
    score: float
    sentence_scores: List[float]           # legacy compatibility mirror
    failing_sentence_indices: List[int]    # legacy compatibility mirror
    instance_refine_counts: List[int]      # legacy compatibility mirror
    sentence_records: List[SentenceRecord]    # per-sentence canonical records
    records_consistency: Dict[str, Any]
    refine_count: Annotated[int, add]

    # existing tooling fields
    news_article: Optional[str]
    news_generation_result: list[str]
    mcp_result: str
    news_hash: Set[str]
    news_dict: Dict[str, Any]


class TopicState(BaseState, total=False):
    task: Literal["topic"]
    label: str
    # task_validation_result: TaskValidationResult


class SentimentState(BaseState, total=False):
    task: Literal["sentiment"]
    label: Literal["positive", "negative", "neutral"]
    task_constraints: Dict[str, Any]
    # task_validation_result: TaskValidationResult


class NERSpan(TypedDict):
    start: int
    end: int
    type: str

class NERState(BaseState, total=False):
    task: Literal["ner"]
    annotations: list[NERSpan]
    task_constraints: Dict[str, Any]
    # task_validation_result: TaskValidationResult


AgentRunningState = BaseState

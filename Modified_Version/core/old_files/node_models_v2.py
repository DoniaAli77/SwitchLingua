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
class TaskValidationResult(TypedDict, total=False):
    passed: bool
    confidence: float
    notes: str
    predicted_label: Optional[str]
    errors: list[str]

class BaseState(TypedDict, total=False):
    task: Literal["topic", "sentiment", "ner"]

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

    # generation outputs
    text: Optional[str]            # new canonical field (recommended)
    data_generation_result: list[str]
    response: str

    # quality results (unchanged types)
    fluency_result: FluencyResponse
    naturalness_result: NaturalnessResponse
    cs_ratio_result: CSRatioResponse
    social_cultural_result: SocialCulturalResponse

    summary: str
    score: float
    refine_count: Annotated[int, add]

    # existing tooling fields
    news_article: Optional[str]
    news_generation_result: list[str]
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
    type: Literal["PER", "ORG", "LOC"]

class NERState(BaseState, total=False):
    task: Literal["ner"]
    annotations: list[NERSpan]
    task_constraints: Dict[str, Any]
    # task_validation_result: TaskValidationResult


AgentRunningState = Union[TopicState, SentimentState, NERState]

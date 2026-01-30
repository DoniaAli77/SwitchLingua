# SwitchLingua: End-to-End Flow

This document explains the complete pipeline from **input configuration** → **scenario generation** → **agent execution** → **scoring** → **output**.

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Configuration & Scenario Generation](#phase-1-configuration--scenario-generation)
3. [Phase 2: Agent Orchestration (StateGraph)](#phase-2-agent-orchestration-stategraph)
4. [Phase 3: Node Execution](#phase-3-node-execution)
5. [Phase 4: Evaluation & Scoring](#phase-4-evaluation--scoring)
6. [Phase 5: Refinement Decision](#phase-5-refinement-decision)
7. [Phase 6: Output & Persistence](#phase-6-output--persistence)
8. [Running the Pipeline](#running-the-pipeline)
9. [Example Trace](#example-trace)

---

## Overview

**SwitchLingua** is a **code-switching data generation and evaluation system** that uses LLMs to create multilingual text samples and then evaluates them across multiple dimensions (fluency, naturalness, ratio correctness, socio-cultural appropriateness).

The system follows this flow:

```
Config YAML
    ↓
[load_config]
    ↓
Scenarios (Cartesian Product)
    ↓
[generate_scenarios]
    ↓
For each scenario:
  ├─ CodeSwitchingAgent
  │  └─ StateGraph (async)
  │     ├─ DataGenerationAgent
  │     ├─ [FluencyAgent || NaturalnessAgent || CSRatioAgent || SocialCulturalAgent]  (parallel)
  │     ├─ SummarizeResult
  │     ├─ meet_criteria (conditional branch)
  │     │  ├─ RefinerAgent (if score < 8 and refine_count < MAX)
  │     │  └─ AcceptanceAgent (otherwise)
  │     └─ END
    ↓
Output JSONL files (one per language)
```

---

## Phase 1: Configuration & Scenario Generation

### Step 1a: Load Configuration

**File:** [`core/utils.py`](core/utils.py#L1-L25)

```python
def load_config(config_path: str):
    """Load a YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
```

**Input:** A YAML config file (e.g., `config/config_augmented_french_eng.yaml`) containing:

```yaml
pre_execute:
  cs_ratio:
    - "70%"
    - "50%"
    - "30%"
  use_tools: true
  topics:
    - "business"
    - "sports"
    - "health"
  tense:
    - "Past"
    - "Present"
    - "Future"
  perspective:
    - "First Person"
    - "Third Person"
  cs_function:
    - "Directive"
    - "Expressive"
    - "Referential"
  cs_type:
    - "Intersentential"
    - "Intrasentential"
    - "Extra-sentential / Tag switching"
  character_setting:
    nationality:
      first_language: "French"
      second_language: "English"
    age:
      - "18-25"
      - "26-35"
      - "56-65"
    gender:
      - "Male"
      - "Female"
    education_level:
      - "High School"
      - "College"
      - "Master"
  conversation_type:
    - "single_turn"
    - "multi-turn"
  output_format: "json"

on_execute:
  round: 1
  verbose: true
```

### Step 1b: Generate Scenarios (Cartesian Product)

**File:** [`core/utils.py`](core/utils.py#L27-L120)

```python
def generate_scenarios(config: dict) -> list[AgentRunningState]:
    """
    Expand config lists into all possible scenario combinations.
    """
    topics = config["topics"]
    tenses = config["tense"]
    perspectives = config["perspective"]
    # ... more fields ...
    
    all_scenarios = []
    for (topic, tense, perspective, ...) in itertools.product(
        topics, tenses, perspectives, ...
    ):
        scenario = {
            "topic": topic,
            "tense": tense,
            "perspective": perspective,
            # ... all other fields ...
            "first_language": config["character_setting"]["nationality"]["first_language"],
            "second_language": config["character_setting"]["nationality"]["second_language"],
        }
        all_scenarios.append(scenario)
    
    return all_scenarios
```

**Output:** A list of scenario dictionaries. Each scenario is an `AgentRunningState` (TypedDict) containing all parameters needed for a single generation run.

**Example Scenario (one of many):**

```python
{
    "topic": "sports",
    "tense": "Present",
    "perspective": "Third Person",
    "cs_ratio": "30%",
    "gender": "Female",
    "age": "26-35",
    "education_level": "College",
    "first_language": "French",
    "second_language": "English",
    "conversation_type": "single_turn",
    "cs_function": "Expressive",
    "cs_type": "Intersentential",
}
```

**Key insight:** The number of scenarios = product of all list lengths. For the config above:
- 3 topics × 3 tenses × 2 perspectives × 3 ages × 2 genders × 3 education levels × 3 cs_ratios × 2 conversation_types × 3 cs_functions × 3 cs_types = **~1,944 scenarios**

---

## Phase 2: Agent Orchestration (StateGraph)

**File:** [`core/agents.py`](core/agents.py#L42-L120)

Each scenario is executed through a **StateGraph** (from `langgraph.graph`), which is a directed acyclic graph (DAG) of nodes and edges.

### Graph Construction

```python
class CodeSwitchingAgent:
    def __init__(self, scenario_k):
        self.state = AgentRunningState()
        # Populate state with scenario data
        for key in scenario_k.keys():
            self.state[key] = scenario_k[key]
        
        self.state["news_article"] = ""
        self.state["news_hash"] = set()
        self.state["news_dict"] = {}
        
        # Construct the graph
        self.workflow_with_data_generation = self._construct_graph_with_data_generation()
    
    def _construct_graph_with_data_generation(self) -> StateGraph:
        workflow = StateGraph(AgentRunningState)
        
        # Add nodes
        workflow.add_node("DataGenerationAgent", RunDataGenerationAgent)
        workflow.add_node("FluencyAgent", RunFluencyAgent)
        workflow.add_node("NaturalnessAgent", RunNaturalnessAgent)
        workflow.add_node("CSRatioAgent", RunCSRatioAgent)
        workflow.add_node("SocialCulturalAgent", RunSocialCulturalAgent)
        workflow.add_node("SummarizeResult", SummarizeResult)
        workflow.add_node("RefinerAgent", RunRefinerAgent)
        workflow.add_node("AcceptanceAgent", AcceptanceAgent)
        
        # Add edges
        workflow.add_edge(START, "DataGenerationAgent")
        workflow.add_edge("DataGenerationAgent", "FluencyAgent")
        workflow.add_edge("DataGenerationAgent", "NaturalnessAgent")
        workflow.add_edge("DataGenerationAgent", "CSRatioAgent")
        workflow.add_edge("DataGenerationAgent", "SocialCulturalAgent")
        
        # All evaluators feed into SummarizeResult
        workflow.add_edge(
            ["FluencyAgent", "NaturalnessAgent", "CSRatioAgent", "SocialCulturalAgent"],
            "SummarizeResult",
        )
        
        # Conditional edge: SummarizeResult → RefinerAgent or AcceptanceAgent
        workflow.add_conditional_edges("SummarizeResult", meet_criteria)
        
        workflow.add_edge("RefinerAgent", "SummarizeResult")  # Loop back
        workflow.add_edge("AcceptanceAgent", END)
        
        return workflow.compile()
```

### Graph Visualization

```
    START
      ↓
DataGenerationAgent
  ├→ FluencyAgent ──────┐
  ├→ NaturalnessAgent ──┤
  ├→ CSRatioAgent ──────┤
  └→ SocialCulturalAgent┤
                        ↓
                  SummarizeResult
                        ↓
                   meet_criteria
                    /         \
                   /           \
          score < 8?           score ≥ 8?
              ↓                   ↓
        RefinerAgent      AcceptanceAgent
              ↓                   ↓
        SummarizeResult          END
              ↓
        (loop back or accept)
```

### Execution

```python
async def run(self):
    return await self.workflow_with_data_generation.ainvoke(
        self.state, {"recursion_limit": 1e10}
    )
```

The StateGraph is executed **asynchronously**, allowing parallel execution of evaluation nodes.

---

## Phase 3: Node Execution

**File:** [`core/node_engine.py`](core/node_engine.py)

Each node is a function that accepts `state: AgentRunningState` and returns a dict to merge into the state.

### 3.1 DataGenerationAgent

**Function:** `RunDataGenerationAgent(state)`

```python
def RunDataGenerationAgent(state: AgentRunningState):
    DataGenerationAgent = DATA_GENERATION_PROMPT | ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        base_url=API_BASE,
        api_key=API_KEY
    ).with_structured_output(GenerationResponse)
    
    response = DataGenerationAgent.invoke(state)
    
    # Retry if no instances generated
    retry = 4
    if not response.get("instances"):
        while retry > 0:
            response = DataGenerationAgent.invoke(state)
            if response.get("instances"):
                break
            retry -= 1
    
    return {"data_generation_result": response["instances"]}
```

**Input State Fields Used:**
- `topic`, `tense`, `perspective`, `first_language`, `second_language`, `cs_ratio`, `cs_function`, `cs_type`, `gender`, `age`, `education_level`, `conversation_type`

**Prompt:** [`core/prompt.py`](core/prompt.py) — `DATA_GENERATION_PROMPT`

**Output State Fields Added:**
- `"data_generation_result"`: list of code-switched strings

**Example Output:**
```python
{
    "data_generation_result": [
        "L'équipe a démarré le match avec force. But then the defense couldn't keep up.",
        "Le public était vraiment enthousiaste au premier quart. The mood shifted quickly.",
        "La défaite était difficile pour tout le monde. It's frustrating to lose after such a promising start."
    ]
}
```

---

### 3.2 Evaluation Agents (Parallel)

All four evaluation agents run **in parallel** after `DataGenerationAgent` completes. Each uses the generated text to produce a structured evaluation response.

#### 3.2.1 FluencyAgent

**Function:** `RunFluencyAgent(state)`

```python
def RunFluencyAgent(state: AgentRunningState):
    FluencyAgent = FLUENCY_PROMPT | ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        base_url=API_BASE,
        api_key=API_KEY
    ).with_structured_output(FluencyResponse)
    
    response = FluencyAgent.invoke(state)
    return {"fluency_result": response}
```

**Input State Fields Used:**
- `data_generation_result` (the generated text)

**Prompt:** [`core/prompt.py`](core/prompt.py) — `FLUENCY_PROMPT`

**Output Type:** `FluencyResponse` (TypedDict)

```python
class FluencyResponse(TypedDict):
    fluency_score: float  # 0-10
    errors: dict[str, str]  # {error_description: constraint_violated}
    summary: str  # Human-readable summary
```

**Example Output:**
```python
{
    "fluency_result": {
        "fluency_score": 9.0,
        "errors": {},
        "summary": "High fluency; sentence-level switches respect Free Morpheme and Equivalence constraints."
    }
}
```

#### 3.2.2 NaturalnessAgent

**Function:** `RunNaturalnessAgent(state)`

Evaluates if the code-switching sounds natural to bilingual speakers (not forced or awkward).

**Output Type:** `NaturalnessResponse`

```python
class NaturalnessResponse(TypedDict):
    naturalness_score: float  # 0-10
    observations: dict[str, str]  # Per-sentence analysis
    summary: str
```

#### 3.2.3 CSRatioAgent

**Function:** `RunCSRatioAgent(state)`

Checks if the matrix/embedded language ratio matches the target (e.g., "30%" embedded English).

**Output Type:** `CSRatioResponse`

```python
class CSRatioResponse(TypedDict):
    ratio_score: float  # 0-10
    computed_ratio: str  # e.g., "63% French : 37% English"
    notes: str
```

#### 3.2.4 SocialCulturalAgent

**Function:** `RunSocialCulturalAgent(state)`

Ensures the text respects cultural norms and uses appropriate expressions (not offensive or unnatural).

**Output Type:** `SocialCulturalResponse`

```python
class SocialCulturalResponse(TypedDict):
    socio_cultural_score: float  # 0-10
    issues: str  # Problems found (if any)
    summary: str
```

---

### 3.3 (Optional) MCPAgent

**File:** [`core/mcp_tools.py`](core/mcp_tools.py)

The MCP (Model Context Protocol) system allows plugin-style tools to be registered and executed.

**Function:** `RunMCPAgent(state)`

```python
def RunMCPAgent(state: AgentRunningState) -> Dict[str, Any]:
    """
    Iterate through all MCP tools in the registry, execute them in order,
    and merge the results into state["mcp_result"].
    """
    result: Dict[str, Any] = {}
    for tool_name, tool in get_all_tools().items():
        try:
            result.update(tool.run(state))
        except Exception as e:
            result[tool_name] = f"ERROR: {e}"
    return {"mcp_result": result}
```

**Example Tool:** `SampleWordCountTool`

```python
@register
class SampleWordCountTool:
    """Count the total number of tokens in data_generation_result"""
    name = "word_count"
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        instances = state.get("data_generation_result", [])
        token_cnt = sum(len(s.split()) for s in instances)
        return {self.name: token_cnt}
```

**Output State Field:**
- `"mcp_result"`: dict of tool results (can be empty if no tools registered or MCP disabled)

---

### 3.4 SummarizeResult Node

**Function:** `SummarizeResult(state)`

```python
def SummarizeResult(state: AgentRunningState):
    summary = f"""
    data_generation_result: {state["data_generation_result"]}
    Fluency Result: {state["fluency_result"]}
    Naturalness Result: {state["naturalness_result"]}
    CSRatio Result: {state["cs_ratio_result"]}
    Social Cultural Result: {state["social_cultural_result"]}
    """
    state["summary"] = summary
    
    return {"score": weighting_scheme(state), "summary": summary}
```

**Purpose:** Combines all evaluator scores into a single weighted score.

---

## Phase 4: Evaluation & Scoring

### Weighting Scheme

**File:** [`core/utils.py`](core/utils.py#L120-L136)

```python
def weighting_scheme(state):
    fluency = state["fluency_result"]["fluency_score"]
    naturalness = state["naturalness_result"]["naturalness_score"]
    csratio = state["cs_ratio_result"]["ratio_score"]
    socio = state["social_cultural_result"]["socio_cultural_score"]
    
    # Weighted average
    return fluency * 0.3 + naturalness * 0.25 + csratio * 0.2 + socio * 0.25
```

**Weights:**
- **Fluency: 30%** — Grammatical correctness and syntactic coherence are most important.
- **Naturalness: 25%** — Does it sound like real bilingual speech?
- **CS Ratio: 20%** — Did it match the target language proportion?
- **Socio-Cultural: 25%** — Is it culturally appropriate?

### Example Score Calculation

Given a state with evaluation results:
```
fluency_score: 9.0
naturalness_score: 8.5
ratio_score: 7.0
socio_cultural_score: 9.0
```

Score = 9.0 × 0.3 + 8.5 × 0.25 + 7.0 × 0.2 + 9.0 × 0.25
       = 2.7 + 2.125 + 1.4 + 2.25
       = **8.475**

---

## Phase 5: Refinement Decision

**File:** [`core/agents.py`](core/agents.py#L1-L30)

```python
def meet_criteria(state: AgentRunningState):
    """Conditional branch: should we refine or accept?"""
    if state["score"] < 8 and state["refine_count"] < MAX_REFINER_ITERATIONS:
        return "RefinerAgent"
    else:
        return "AcceptanceAgent"

MAX_REFINER_ITERATIONS = 1
```

### Logic

- **If score < 8 AND refine_count < MAX_REFINER_ITERATIONS:**
  - Route to `RefinerAgent` to attempt improvement.
  - `RefinerAgent` increments `refine_count` and re-generates text.
  - Graph loops back to evaluators.

- **Else (score ≥ 8 OR max refinements reached):**
  - Route to `AcceptanceAgent` to save output.

### RefinerAgent

**Function:** `RunRefinerAgent(state)`

```python
def RunRefinerAgent(state: AgentRunningState):
    RefinerAgent = REFINER_PROMPT | ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        base_url=API_BASE,
        api_key=API_KEY
    ).with_structured_output(GenerationResponse)
    
    response = RefinerAgent.invoke(state)
    
    return {"refiner_result": response, "refine_count": 3}
```

The refiner receives the evaluation summary and attempts to fix identified issues.

---

## Phase 6: Output & Persistence

**Function:** `AcceptanceAgent(state)`

```python
def AcceptanceAgent(state: AgentRunningState):
    # Remove transient fields
    del state["news_article"]
    del state["news_hash"]
    del state["news_dict"]
    
    # Get language identifier
    language = state["first_language"]
    
    # Append to language-specific JSONL file
    with jsonlines.open(
        f"{OUTPUT_DIR}/{language}.jsonl",
        "a",
    ) as f:
        f.write(state)
    
    return
```

### Output Structure

**File:** `output/{language}.jsonl` (one file per first_language)

**Format:** JSON Lines (newline-delimited JSON)

**Each line contains:**
```json
{
  "topic": "sports",
  "tense": "Present",
  "perspective": "Third Person",
  "cs_ratio": "30%",
  "gender": "Female",
  "age": "26-35",
  "education_level": "College",
  "first_language": "French",
  "second_language": "English",
  "conversation_type": "single_turn",
  "cs_function": "Expressive",
  "cs_type": "Intersentential",
  "data_generation_result": [
    "L'équipe a démarré le match avec force. But then the defense couldn't keep up and things started to fall apart.",
    "Le public était vraiment enthousiaste au premier quart. The mood shifted quickly after halftime."
  ],
  "fluency_result": {
    "fluency_score": 9.0,
    "errors": {},
    "summary": "High fluency; sentence-level switches are natural."
  },
  "naturalness_result": {
    "naturalness_score": 8.5,
    "observations": {},
    "summary": "Sounds natural for bilingual speakers."
  },
  "cs_ratio_result": {
    "ratio_score": 7.0,
    "computed_ratio": "63% French : 37% English",
    "notes": "Slightly higher embedded language usage than target."
  },
  "social_cultural_result": {
    "socio_cultural_score": 9.0,
    "issues": "",
    "summary": "No cultural problems detected."
  },
  "summary": "data_generation_result: [...] Fluency Result: {...} ...",
  "score": 8.475,
  "refine_count": 0
}
```

---

## Running the Pipeline

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```powershell
   setx API_KEY "your_api_key_here"
   setx API_BASE "https://api.your-provider.com"
   ```

3. **Create/configure a config file** (e.g., `config/config_augmented_french_eng.yaml`)

4. **Update `OUTPUT_DIR` in `core/node_engine.py`:**
   ```python
   OUTPUT_DIR = "output"  # or a path of your choice
   ```

### Run the Pipeline

**Option A: Full pipeline (with LLM calls)**

```bash
python core/run_french.py
# or
python core/agents.py
```

**Option B: Mock run (no LLM calls, for testing structure)**

```bash
python core/mock_run.py
```

This will:
1. Create a fake scenario state.
2. Run any registered MCP tools.
3. Compute a weighted score.
4. Write output to `output/mock_run_result.jsonl`.

---

## Example Trace

### Scenario Input (from config expansion)

```python
{
    "topic": "sports",
    "tense": "Present",
    "perspective": "Third Person",
    "cs_ratio": "30%",
    "gender": "Female",
    "age": "26-35",
    "education_level": "College",
    "first_language": "Arabic",
    "second_language": "English",
    "conversation_type": "single_turn",
    "cs_function": "Expressive",
    "cs_type": "Intersentential",
}
```

### Step 1: DataGenerationAgent runs

LLM prompt with all scenario fields → generates 3 code-switched sentences:

```
data_generation_result: [
    "الفريق بدأ المباراة بشكل قوي جدًا. But then the defense couldn't keep up.",
    "الجمهور كان متحمس جدًا. The mood shifted quickly after halftime.",
    "الخسارة كانت صعبة على الجميع. It's frustrating to lose after a promising start."
]
```

### Step 2: Evaluation Agents run in parallel

- **FluencyAgent:** score=9.0 (no grammatical errors, respects constraints)
- **NaturalnessAgent:** score=8.5 (sounds natural, inter-sentential switches are common)
- **CSRatioAgent:** score=7.0 (computed 63% Arabic : 37% English, target was 30%)
- **SocialCulturalAgent:** score=9.0 (no offensive content, idiomatic in both languages)

### Step 3: SummarizeResult computes score

```
score = 9.0 × 0.3 + 8.5 × 0.25 + 7.0 × 0.2 + 9.0 × 0.25
      = 2.7 + 2.125 + 1.4 + 2.25
      = 8.475
```

### Step 4: Conditional decision

- `score` = 8.475
- `score < 8`? **No** → route to `AcceptanceAgent`

### Step 5: AcceptanceAgent writes output

The full state (all fields + scores + evaluations) is appended to `output/Arabic.jsonl`.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `core/utils.py` | Config loading, scenario generation, weighting scheme |
| `core/agents.py` | `CodeSwitchingAgent` class, StateGraph construction, `main()` entry point |
| `core/node_engine.py` | All node functions (DataGen, Fluency, Naturalness, CSRatio, SocialCultural, MCP, Summarize, Acceptance, Refiner) |
| `core/node_models.py` | TypedDict definitions (state, response types) |
| `core/prompt.py` | All LLM prompts (DATA_GENERATION, FLUENCY, NATURALNESS, CS_RATIO, SOCIAL_CULTURAL, REFINER) |
| `core/mcp_tools.py` | MCP tool registry and example `SampleWordCountTool` |
| `core/mock_run.py` | Mock runner (no LLM calls, for testing) |
| `core/run_french.py` | Variant orchestrator with MCP node and webhook integration |

---

## Summary

1. **Config → Scenarios:** YAML config is expanded via Cartesian product into thousands of scenario dicts.
2. **Scenario → StateGraph:** Each scenario populates an `AgentRunningState` and is executed through a compiled StateGraph.
3. **StateGraph Execution:** Nodes run in sequence/parallel:
   - DataGenerationAgent generates code-switched text.
   - Evaluation agents (Fluency, Naturalness, CSRatio, SocialCultural) score the output in parallel.
   - SummarizeResult combines scores via `weighting_scheme()`.
   - Conditional branch decides refine or accept.
4. **Refinement Loop:** If score < threshold and attempts remain, RefinerAgent improves the text and evaluators re-run.
5. **Output:** Accepted states are persisted to language-specific JSONL files.

---

**For more details, refer to the source code in `core/` and adjust configuration/weights as needed for your use case.**

# Scoring Clarification: Per Scenario vs Per Sentence

## Quick Answer

**Agents score per scenario**, not per individual sentence.

---

## What is a "Scenario"?

A **scenario** is one complete generation run with:
- One set of parameters (topic, language pair, tense, etc.)
- One generated output: **3–N code-switched sentences** (a list called `data_generation_result`)
- One set of evaluation scores (one fluency score, one naturalness score, etc. for the entire list)

---

## Data Flow

### Input to Evaluators

```python
# Scenario state
state = {
    "topic": "sports",
    "first_language": "Arabic",
    "second_language": "English",
    "cs_ratio": "30%",
    "tense": "Present",
    # ... other fields ...
    
    # Generated output (list of sentences)
    "data_generation_result": [
        "الفريق بدأ المباراة بشكل قوي جدًا وحقق تقدم كبير في البداية. But then the defense couldn't keep up and things started to fall apart.",
        "الجمهور كان متحمس جدًا في الربع الأول. The mood shifted quickly after halftime when the game got tense.",
        "الخسارة كانت صعبة على الجميع، خصوصًا بعد الأداء القوي في البداية. It's frustrating to see them lose after such a promising start."
    ]
}
```

### Evaluator Execution

```python
def RunFluencyAgent(state: AgentRunningState):
    # Note: receives the ENTIRE state, including the full list
    FluencyAgent = FLUENCY_PROMPT | ChatOpenAI(...)
    
    response = FluencyAgent.invoke(state)
    # Input to LLM: ALL 3 sentences at once
    # Output: ONE fluency_score for the entire batch
    
    return {"fluency_result": response}
```

### Output from Evaluators

```python
# ONE score per scenario, not per sentence
"fluency_result": {
    "fluency_score": 9.0,  # ← ONE score for all 3 sentences together
    "errors": [],
    "summary": "All switches occur at clause or sentence boundaries, respecting constraints."
}

"naturalness_result": {
    "naturalness_score": 8.5,  # ← ONE score for all 3 sentences together
    "observations": {
        "sentence_1": "The switch from Arabic to English occurs at a sentence boundary, which is common and natural.",
        "sentence_2": "Again, the switch is at a sentence boundary. Both parts are natural.",
        "sentence_3": "The switch is at a sentence boundary. Both parts are natural."
    },
    "summary": "The code-switching in these sentences is highly natural."
}
```

---

## Why Per-Scenario Scoring?

### Reason 1: Holistic Evaluation
Code-switching quality is not just about individual sentences—it's about **overall coherence** and **consistent naturalness** across the dialogue.

### Reason 2: Prompt Design
The evaluator prompts receive the **full list** (`data_generation_result`) and are instructed to evaluate the **entire batch**:

From `core/prompt.py`:

```python
FLUENCY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "assistant",
        """
        You are **FluencyAgent**. Your task is to evaluate the grammatical 
        correctness and syntactic coherence of code-switched text.
        
        [Fluency criteria...]
        
        Output:
        - A `fluency_score` (0 to 10).
        - A list of identified `errors` (if any).
        - A short `summary` of overall fluency.
        
        given the code-switched text {data_generation_result}.
        """
    )
])
```

**Key phrase:** "given the code-switched text {data_generation_result}" — the **entire list** is passed.

### Reason 3: Computational Efficiency
Evaluating 3 sentences as a batch (1 API call) is more efficient than evaluating each sentence individually (3 API calls).

---

## Example: What the LLM Sees

**Prompt passed to LLM:**

```
You are FluencyAgent. Your task is to evaluate the grammatical correctness 
and syntactic coherence of code-switched text.

[Criteria...]

given the code-switched text:
1. الفريق بدأ المباراة بشكل قوي جدًا وحقق تقدم كبير في البداية. But then the defense couldn't keep up and things started to fall apart.
2. الجمهور كان متحمس جدًا في الربع الأول. The mood shifted quickly after halftime when the game got tense.
3. الخسارة كانت صعبة على الجميع، خصوصًا بعد الأداء القوي في البداية. It's frustrating to see them lose after such a promising start.

Evaluate the overall fluency of this code-switched text and provide:
- A fluency_score (0 to 10)
- Identified errors (if any)
- A summary
```

**LLM responds:**

```json
{
  "fluency_score": 9.0,
  "errors": {},
  "summary": "All 3 sentences demonstrate high fluency. The code-switches occur at clause/sentence boundaries, respecting both the Free Morpheme Constraint and the Equivalence Constraint. No grammatical errors detected."
}
```

**Key point:** The LLM gave **one 9.0 score** for the **entire batch of 3 sentences**, not three separate scores.

---

## Scoring Hierarchy

```
┌─────────────────────────────────────────┐
│        SCENARIO (one generation run)    │
│  Parameters: topic, languages, etc.     │
│  Output: 3 code-switched sentences      │
└─────────────────────────────────────────┘
                    ↓
    ┌───────────────┬───────────────┬───────────────┐
    ↓               ↓               ↓               ↓
[Fluency]    [Naturalness]    [CSRatio]    [SocioCultural]
  9.0              8.5            7.0              9.0
(1 score)      (1 score)       (1 score)       (1 score)
    ↓               ↓               ↓               ↓
    └───────────────┴───────────────┴───────────────┘
                    ↓
          ┌──────────────────────┐
          │   Weighted Average   │
          │  (weighting_scheme)  │
          │      = 8.475         │
          └──────────────────────┘
                    ↓
          One FINAL SCORE per scenario
```

---

## Where Per-Sentence Info Appears

Although **overall scores are per-scenario**, some evaluators provide **per-sentence observations**:

```python
"naturalness_result": {
    "naturalness_score": 8.5,  # ← ONE overall score
    "observations": {
        "sentence_1": "The switch from Arabic to English occurs at a sentence boundary...",
        "sentence_2": "Again, the switch is at a sentence boundary...",
        "sentence_3": "The switch is at a sentence boundary..."
    },
    "summary": "The code-switching in these sentences is highly natural."
}
```

**Note:** The observations break down the reasoning, but the **score (8.5) is a single aggregate** for all sentences combined.

---

## Impact on Output

### Per Scenario
One `.jsonl` record per scenario contains:

```json
{
  "topic": "sports",
  "first_language": "Arabic",
  "second_language": "English",
  "data_generation_result": [3 sentences],
  "fluency_result": {
    "fluency_score": 9.0,
    ...
  },
  "naturalness_result": {
    "naturalness_score": 8.5,
    ...
  },
  "cs_ratio_result": {
    "ratio_score": 7.0,
    ...
  },
  "social_cultural_result": {
    "socio_cultural_score": 9.0,
    ...
  },
  "score": 8.475,
  "refine_count": 0
}
```

**One line = one scenario = one set of evaluation scores.**

---

## Refinement: What Gets Re-evaluated?

When `RefinerAgent` runs:

1. It receives the **entire scenario state** (all 3 sentences + all evaluations)
2. It generates **new code-switched text** (replacing the old 3 sentences)
3. The evaluators re-run on the **new batch** (not the old one)
4. New scores are computed for the **entire new batch**

```
Original scenario:
  data_generation_result: [sent_1, sent_2, sent_3]
  fluency_score: 8.2
  ↓ (score too low, trigger refiner)
  ↓
RefinerAgent improves the text:
  data_generation_result: [sent_1_refined, sent_2_refined, sent_3_refined]
  ↓
Evaluators re-run on the ENTIRE refined batch:
  fluency_score: 8.9  (new overall score for the refined batch)
```

---

## Summary

| Aspect | Answer |
|--------|--------|
| **Scoring granularity** | Per scenario (one scenario = 3–N sentences evaluated together) |
| **One fluency score covers...** | All sentences in that scenario's `data_generation_result` |
| **How many LLM calls per evaluator?** | 1 (for the entire batch) |
| **How many output records per scenario?** | 1 JSONL line with all fields + scores |
| **Are sentences scored individually?** | No, but evaluators may provide per-sentence observations for reasoning |
| **What triggers re-evaluation?** | Refinement of the entire batch; evaluators run again on new sentences |

---

## Visual Example

**Config expands to 1,944 scenarios.**

For each scenario:

```
Scenario #1:
  Parameters: {topic: sports, lang_pair: Arabic-English, ...}
  Generate: [sent_1, sent_2, sent_3]
  Evaluate: 1 fluency score, 1 naturalness score, 1 ratio score, 1 socio score
  Output: 1 JSONL record

Scenario #2:
  Parameters: {topic: business, lang_pair: Arabic-English, ...}
  Generate: [sent_1, sent_2, sent_3]
  Evaluate: 1 fluency score, 1 naturalness score, 1 ratio score, 1 socio score
  Output: 1 JSONL record

...

Scenario #1944:
  Parameters: {topic: health, lang_pair: Arabic-English, ...}
  Generate: [sent_1, sent_2, sent_3]
  Evaluate: 1 fluency score, 1 naturalness score, 1 ratio score, 1 socio score
  Output: 1 JSONL record

Total output: 1,944 JSONL records (one per scenario)
```

---

## Code Reference

**Where this happens:**

- **Scenario generation:** [`core/utils.py`](core/utils.py#L27-L120) — `generate_scenarios()` returns ~1,944 scenario dicts
- **Evaluation with full batch:** [`core/node_engine.py`](core/node_engine.py#L112-L170) — evaluators receive entire `state` (which contains `data_generation_result` as a list)
- **One score per scenario:** [`core/utils.py`](core/utils.py#L120-L136) — `weighting_scheme()` computes one final score per scenario
- **Output (one per scenario):** [`core/node_engine.py`](core/node_engine.py#L170-L200) — `AcceptanceAgent` writes one JSONL line per scenario

---

Hope this clears up the confusion! Feel free to ask if you'd like to dig deeper into any part.

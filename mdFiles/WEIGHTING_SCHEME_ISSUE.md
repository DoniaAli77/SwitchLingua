# The Weighting Scheme Permissiveness Issue

## The Problem You Identified

A scenario with **fully Arabic sentences** (or mostly one language, violating code-switching) can still be **accepted** even though the CS ratio agent correctly detects this problem.

**Why?** The weighting scheme uses a **weighted average**, not **hard constraints**.

---

## Example: How a "Bad" Scenario Gets Accepted

### Scenario State After Evaluation

```python
state = {
    "data_generation_result": [
        "الفريق بدأ المباراة بشكل قوي جدًا وحقق تقدم كبير في البداية.",
        "الجمهور كان متحمس جدًا في الربع الأول.",
        "الخسارة كانت صعبة على الجميع."
        # ↑ ALL ARABIC - NO ENGLISH!
    ],
    "fluency_result": {
        "fluency_score": 9.0,  # Good grammar in Arabic
        "errors": {},
        "summary": "No grammatical errors, sentences are well-formed."
    },
    "naturalness_result": {
        "naturalness_score": 9.0,  # Sounds natural (as pure Arabic)
        "observations": {},
        "summary": "Natural Arabic speech patterns."
    },
    "cs_ratio_result": {
        "ratio_score": 2.0,  # ← TERRIBLE! No code-switching detected
        "computed_ratio": "100% Arabic : 0% English",
        "notes": "Expected 30% English but got 0%. Complete failure of CS requirement."
    },
    "social_cultural_result": {
        "socio_cultural_score": 9.0,  # Culturally appropriate
        "issues": "",
        "summary": "No cultural issues."
    }
}
```

### Score Calculation

```python
score = fluency * 0.3 + naturalness * 0.25 + csratio * 0.2 + socio * 0.25
      = 9.0 * 0.3 + 9.0 * 0.25 + 2.0 * 0.2 + 9.0 * 0.25
      = 2.7 + 2.25 + 0.4 + 2.25
      = 7.6
```

### Refinement Decision

```python
def meet_criteria(state: AgentRunningState):
    if state["score"] < 8 and state["refine_count"] < MAX_REFINER_ITERATIONS:
        return "RefinerAgent"  # ← Goes to refiner (7.6 < 8)
    else:
        return "AcceptanceAgent"

MAX_REFINER_ITERATIONS = 1
```

Score 7.6 < 8, so **RefinerAgent runs once** to try fixing the CS ratio.

### Problem: Refiner Still Fails

The refiner prompt receives the evaluation summary and tries to improve. But if:
1. The LLM struggles to generate actual code-switching (model limitation)
2. The refiner prompt isn't strong enough
3. Random variation causes another mostly-Arabic batch

Then after refinement, the scores might be:
- fluency: 9.0
- naturalness: 9.0
- cs_ratio: 3.0 (still mostly Arabic)
- socio: 9.0
- Final score: 7.75 (still < 8)

**But now `refine_count` has hit `MAX_REFINER_ITERATIONS` (which is 1)**, so the next decision is:

```python
if state["score"] < 8 and state["refine_count"] < MAX_REFINER_ITERATIONS:
    # 7.75 < 8 BUT refine_count (1) is NOT < MAX_REFINER_ITERATIONS (1)
    # Condition is FALSE → goes to AcceptanceAgent
    return "AcceptanceAgent"  # ← ACCEPTED despite poor CS ratio!
else:
    return "AcceptanceAgent"
```

### Result

**The scenario is ACCEPTED even though it has 0% code-switching!** ✗

---

## Why This Happens: Design Philosophy

The system uses a **weighted average** (not hard constraints) because:

### 1. Trade-offs are Intentional
Some scenarios might legitimately:
- Have lower CS ratio if the code-switching is very natural and fluent
- Have lower naturalness if the fluency is extremely high
- The system allows trading off one dimension for another

### 2. No Hard Reject Rules
There's no logic like:
```python
# This DOESN'T exist in the codebase:
if state["cs_ratio_result"]["ratio_score"] < 5:
    return "AcceptanceAgent"  # Hard reject on CS ratio
```

### 3. Limited Refinement Iterations
```python
MAX_REFINER_ITERATIONS = 1  # Only 1 attempt to fix!
```

If refinement fails, the scenario is accepted regardless of score. This is:
- **Good for:** Speed/cost (don't refine forever)
- **Bad for:** Quality enforcement (scenarios can slip through)

---

## The Weighting Scheme Problem

### Current Weights

```python
score = fluency * 0.3 + naturalness * 0.25 + csratio * 0.2 + socio * 0.25
```

**Issue:** If one dimension (e.g., CS ratio) is critical, **20% weight is not enough** to block a scenario.

### Example: Two Extremes

**Scenario A (Good CS, Good Fluency):**
```
fluency: 8.0, naturalness: 8.0, csratio: 8.0, socio: 8.0
score = 8.0 (all dimensions balanced)
```

**Scenario B (Bad CS, Excellent Everything Else):**
```
fluency: 9.5, naturalness: 9.5, csratio: 1.0, socio: 9.5
score = 9.5 * 0.3 + 9.5 * 0.25 + 1.0 * 0.2 + 9.5 * 0.25
      = 2.85 + 2.375 + 0.2 + 2.375
      = 7.8
```

Scenario B still gets refined once, but if it fails refinement, **it's accepted despite having 99% one language!**

---

## Solutions (If This Is a Problem)

### Option 1: Increase CS Ratio Weight

```python
# core/utils.py
def weighting_scheme(state):
    fluency = state["fluency_result"]["fluency_score"]
    naturalness = state["naturalness_result"]["naturalness_score"]
    csratio = state["cs_ratio_result"]["ratio_score"]
    socio = state["social_cultural_result"]["socio_cultural_score"]
    
    # New weights: CS ratio is now 40% (more critical)
    return fluency * 0.2 + naturalness * 0.2 + csratio * 0.4 + socio * 0.2
```

With this change, Scenario B would score:
```
9.5 * 0.2 + 9.5 * 0.2 + 1.0 * 0.4 + 9.5 * 0.2
= 1.9 + 1.9 + 0.4 + 1.9
= 6.1  (much lower, more likely to refine)
```

### Option 2: Hard Constraint on CS Ratio

```python
# core/agents.py
def meet_criteria(state: AgentRunningState):
    # Hard constraint: CS ratio must be at least 4.0
    if state["cs_ratio_result"]["ratio_score"] < 4:
        return "RefinerAgent"
    
    # Otherwise use normal logic
    if state["score"] < 8 and state["refine_count"] < MAX_REFINER_ITERATIONS:
        return "RefinerAgent"
    else:
        return "AcceptanceAgent"
```

This ensures scenarios with terrible CS ratio always get refined (at least once).

### Option 3: Increase Max Refinement Iterations

```python
# core/agents.py
MAX_REFINER_ITERATIONS = 3  # Try up to 3 times instead of 1
```

Gives the refiner more chances to fix the CS ratio issue.

### Option 4: Fail Fast if Any Score < Threshold

```python
# core/agents.py
def meet_criteria(state: AgentRunningState):
    # If ANY dimension is critically low, refine
    min_score = min(
        state["fluency_result"]["fluency_score"],
        state["naturalness_result"]["naturalness_score"],
        state["cs_ratio_result"]["ratio_score"],
        state["social_cultural_result"]["socio_cultural_score"],
    )
    
    if min_score < 5 and state["refine_count"] < MAX_REFINER_ITERATIONS:
        return "RefinerAgent"
    
    if state["score"] < 8 and state["refine_count"] < MAX_REFINER_ITERATIONS:
        return "RefinerAgent"
    
    return "AcceptanceAgent"
```

This ensures no single dimension can be ignored.

---

## What's Currently Happening in Your Repo

**File:** [`core/agents.py`](core/agents.py#L1-L30) and [`core/node_engine.py`](core/node_engine.py#L120-L136)

```python
# Current logic
def meet_criteria(state: AgentRunningState):
    if state["score"] < 8 and state["refine_count"] < MAX_REFINER_ITERATIONS:
        return "RefinerAgent"
    else:
        return "AcceptanceAgent"

MAX_REFINER_ITERATIONS = 1

# Current weighting
def weighting_scheme(state):
    fluency = state["fluency_result"]["fluency_score"]
    naturalness = state["naturalness_result"]["naturalness_score"]
    csratio = state["cs_ratio_result"]["ratio_score"]
    socio = state["social_cultural_result"]["socio_cultural_score"]
    
    return fluency * 0.3 + naturalness * 0.25 + csratio * 0.2 + socio * 0.25
```

**Current behavior:**
- ✓ Will refine once if overall score < 8
- ✗ Will NOT refine if overall score ≥ 8, **even if CS ratio is 1.0**
- ✗ Will accept any scenario after 1 refinement attempt, regardless of dimension scores

---

## Recommendation

If **code-switching quality is non-negotiable**, implement **Option 4** (fail fast on any low dimension) plus **Option 1** (increase CS ratio weight):

```python
# core/agents.py
MAX_REFINER_ITERATIONS = 2

def meet_criteria(state: AgentRunningState):
    # Hard check: no dimension should be critically low
    min_score = min(
        state["fluency_result"]["fluency_score"],
        state["naturalness_result"]["naturalness_score"],
        state["cs_ratio_result"]["ratio_score"],
        state["social_cultural_result"]["socio_cultural_score"],
    )
    
    if min_score < 5 and state["refine_count"] < MAX_REFINER_ITERATIONS:
        return "RefinerAgent"
    
    if state["score"] < 8 and state["refine_count"] < MAX_REFINER_ITERATIONS:
        return "RefinerAgent"
    
    return "AcceptanceAgent"

# core/utils.py
def weighting_scheme(state):
    fluency = state["fluency_result"]["fluency_score"]
    naturalness = state["naturalness_result"]["naturalness_score"]
    csratio = state["cs_ratio_result"]["ratio_score"]
    socio = state["social_cultural_result"]["socio_cultural_score"]
    
    # Increased CS ratio weight to 35%
    return fluency * 0.25 + naturalness * 0.2 + csratio * 0.35 + socio * 0.2
```

---

## Summary

You've identified a real **design choice**: the system **trades off one dimension for another** rather than enforcing hard rules. This is:

| Aspect | Impact |
|--------|--------|
| **A scenario with 0% code-switching** | Can still be accepted if fluency/naturalness/socio are very high |
| **CS ratio has only 20% weight** | Not enough to block high-scoring scenarios on other dimensions |
| **Only 1 refinement attempt** | If refiner fails, scenario is accepted regardless |
| **No hard constraint per dimension** | No "minimum score per evaluator" rule |

**This is likely intentional** (allow trade-offs) but might not match your quality expectations. Adjust the weights and constraints above if stricter CS quality is needed.

---
